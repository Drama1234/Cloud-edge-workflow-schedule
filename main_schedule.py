import math
import time
import json
import os
import argparse
import csv
import resource
import torch
import numpy as np
import pickle
from typing import Any, Dict, Tuple, List, Optional
from collections import defaultdict

from env.environment import Environment
from utils.reason import FailureReason
from trajectories.trajectory import Trajectory, TrajectoryCollector
from env.nodes import Node, Master, Cloud, Docker
from data.tasks import Task, Workflow 
from utils.utils import set_seed, ReplayBuffer, get_latency, get_latency_with_match, map_task_to_docker_type, host_id_to_index, DOCKER_CONFIG, NUM_DOCKER_TYPES
from method.cmmac import GlobalReplayBuffer,Actor,Critic,DistributedAgent,CMMACAlgorithmAdapter
from schedule.edgeagent import EdgeAgent

from processed_task_loader import get_all_task
from utils.reward import calculate_reward
from env.env_loader import load_env_from_json

print("[DEBUG] main_schedule.py imported", flush=True)


def _append_metrics_csv(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _path_size_bytes(path: Optional[str]) -> int:
    if not path:
        return 0
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return int(os.path.getsize(path))
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += int(os.path.getsize(fp))
            except OSError:
                continue
    return int(total)


def _module_param_stats(module: Any) -> Tuple[int, int]:
    if module is None:
        return 0, 0
    if not isinstance(module, torch.nn.Module):
        return 0, 0
    n_params = 0
    n_bytes = 0
    for p in module.parameters():
        try:
            n = int(p.numel())
            n_params += n
            n_bytes += int(n) * int(p.element_size())
        except Exception:
            continue
    return int(n_params), int(n_bytes)


def _sum_param_stats(modules: List[Any]) -> Tuple[int, int]:
    total_params = 0
    total_bytes = 0
    for m in modules:
        p, b = _module_param_stats(m)
        total_params += int(p)
        total_bytes += int(b)
    return int(total_params), int(total_bytes)


def _max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _max_cuda_mem_bytes() -> int:
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.max_memory_allocated())
        except Exception:
            return 0
    return 0


def _reset_cuda_peak_mem() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            return


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            return


def _avg_workflow_makespan(env: Environment) -> Tuple[Optional[float], int]:
    vals = []
    for m in env.masters:
        for wf in m.workflows.values():
            if not getattr(wf, "is_completed", False):
                continue
            ms = wf.compute_makespan()
            if ms is None:
                continue
            vals.append(float(ms))
    if not vals:
        return None, 0
    return float(np.mean(vals)), int(len(vals))


def _latency_pair_match_stats(env: Environment) -> Dict[str, Any]:
    host_ids = [h.id for h in env.all_hosts]
    total = 0
    unmatched = 0
    for i in range(len(host_ids)):
        for j in range(len(host_ids)):
            if i == j:
                continue
            total += 1
            _lat_s, matched = get_latency_with_match(host_ids[i], host_ids[j], env.links)
            if not matched:
                unmatched += 1
    ratio = (float(unmatched) / float(total)) if total > 0 else None
    return {
        "latency_pairs_total": int(total),
        "latency_pairs_unmatched": int(unmatched),
        "latency_pairs_unmatched_ratio": ratio,
    }


def evaluate_du_policy(
        env: Environment,
        cluster_agents: List[EdgeAgent],
        du_policy: Any,
        eval_episodes: int = 1,
        max_env_steps: int = 300000,
        event_driven: bool = True,
) -> Dict[str, Any]:
    eval_episodes = int(eval_episodes)
    max_env_steps = int(max_env_steps)

    all_episode_stats: List[Dict[str, Any]] = []
    inference_target = 1000

    latency_diag = _latency_pair_match_stats(env)

    for ep in range(eval_episodes):
        env.reset()
        if hasattr(du_policy, "reset"):
            du_policy.reset()

        _reset_cuda_peak_mem()

        valid_len_hist: Dict[int, int] = defaultdict(int)
        valid_len_count = 0
        valid_len_sum = 0
        valid_len_min: Optional[int] = None
        valid_len_max: Optional[int] = None
        valid_len_le_1 = 0
        valid_len_le_2 = 0
        valid_len_le_3 = 0
        valid_len_le_5 = 0
        valid_len_le_10 = 0
        cloud_ids = {h.id for h in env.all_hosts if getattr(h, "is_cloud_node", False)}
        cloud_in_valid = 0
        only_cloud_valid = 0
        no_valid_host_steps = 0

        stats: Dict[str, Any] = {
            "episode": ep + 1,
            "env_steps": 0,
            "action_steps": 0,
            "success_actions": 0,
            "failed_actions": 0,
            "local_scheduled": 0,
            "remote_scheduled": 0,
            "cloud_scheduled": 0,
            "total_reward": 0.0,
            "no_action_env_steps": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "total_workflows": env.global_stats.get("total_workflows", 0),
            "diagnostics": {},
        }

        env_steps = 0
        no_action_env_steps = 0
        infer_count = 0
        infer_time = 0.0

        while env_steps < max_env_steps:
            all_terminated = True
            for master in env.masters:
                if not master.is_all_workflows_terminated():
                    all_terminated = False
                    break
            if all_terminated:
                break

            executed_any_action = False

            for master_idx, agent in enumerate(cluster_agents):
                master = env.masters[master_idx]
                picked_wf_id = None
                picked_task = None
                picked_valid_hosts = None
                picked_valid_host_ids = None
                pending_tasks = list(getattr(master, "pending_tasks", []) or [])

                for task in pending_tasks:
                    valid_hosts = agent.get_valid_hosts_for_task(task)
                    valid_host_ids = [h["host_id"] for h in valid_hosts]
                    if not valid_host_ids:
                        continue
                    picked_task = task
                    picked_valid_hosts = valid_hosts
                    picked_valid_host_ids = valid_host_ids
                    picked_wf_id = getattr(task, "workflow_id", None) or getattr(getattr(task, "workflow", None), "id", None)
                    break

                if picked_task is None or picked_wf_id is None:
                    if pending_tasks:
                        no_valid_host_steps += 1
                    continue

                state = env.get_cluster_state(master_idx, focus_task=picked_task)
                valid_hosts = picked_valid_hosts
                valid_host_ids = picked_valid_host_ids

                vlen = int(len(valid_host_ids))
                valid_len_hist[vlen] += 1
                valid_len_count += 1
                valid_len_sum += vlen
                if valid_len_min is None or vlen < valid_len_min:
                    valid_len_min = vlen
                if valid_len_max is None or vlen > valid_len_max:
                    valid_len_max = vlen
                if vlen <= 1:
                    valid_len_le_1 += 1
                if vlen <= 2:
                    valid_len_le_2 += 1
                if vlen <= 3:
                    valid_len_le_3 += 1
                if vlen <= 5:
                    valid_len_le_5 += 1
                if vlen <= 10:
                    valid_len_le_10 += 1
                if cloud_ids and any((cid in valid_host_ids) for cid in cloud_ids):
                    cloud_in_valid += 1
                    if vlen == 1:
                        only_cloud_valid += 1

                ctx_key = f"{master_idx}:{picked_wf_id}"
                if infer_count < inference_target:
                    _sync_cuda()
                    t0 = time.perf_counter()

                chosen_host_id, _ = du_policy.select_action(
                    state=state,
                    valid_host_ids=valid_host_ids,
                    master_idx=ctx_key,
                    epsilon=0.0,
                )

                if infer_count < inference_target:
                    _sync_cuda()
                    infer_time += float(time.perf_counter() - t0)
                    infer_count += 1
                if not chosen_host_id:
                    continue

                success, _ = env.execute_action(master_idx, picked_task.task_id, chosen_host_id)
                reward = agent._calculate_reward(picked_task, success, chosen_host_id, valid_hosts)
                if hasattr(du_policy, "record_reward"):
                    du_policy.record_reward(ctx_key, reward)

                executed_any_action = True
                stats["action_steps"] += 1
                stats["total_reward"] += float(reward)

                if success:
                    stats["success_actions"] += 1
                    host = next(h for h in env.all_hosts if h.id == chosen_host_id)
                    if host in agent.local_hosts:
                        stats["local_scheduled"] += 1
                    elif getattr(host, "is_cloud_node", False):
                        stats["cloud_scheduled"] += 1
                    else:
                        stats["remote_scheduled"] += 1
                else:
                    stats["failed_actions"] += 1

            completed_workflows, failed_workflows = env.step(event_driven=event_driven)
            env_steps += 1
            stats["env_steps"] = env_steps
            stats["completed_workflows"] = int(env.global_stats.get("completed_workflows", 0))
            stats["failed_workflows"] = int(env.global_stats.get("failed_workflows", 0))

            if env_steps % 50 == 0:
                print(
                    f"[eval_du] ep={ep+1}/{eval_episodes} env_steps={env_steps} "
                    f"action_steps={stats['action_steps']} completed={stats['completed_workflows']}/"
                    f"{stats['total_workflows']} failed={stats['failed_workflows']} cur_time={env.cur_time:.2f}",
                    flush=True,
                )

            if not executed_any_action:
                no_action_env_steps += 1
                stats["no_action_env_steps"] = no_action_env_steps
            else:
                no_action_env_steps = 0
                stats["no_action_env_steps"] = 0

        stats["end_time"] = float(env.cur_time)
        stats["global_end_time"] = float(env.cur_time)
        avg_ms, avg_ms_n = _avg_workflow_makespan(env)
        stats["avg_workflow_makespan"] = avg_ms
        stats["avg_workflow_makespan_n"] = int(avg_ms_n)
        stats["inference_decisions_counted"] = int(infer_count)
        stats["inference_1000_decisions_sec"] = float(infer_time) if infer_count >= inference_target else None
        stats["max_rss_kb"] = _max_rss_kb()
        stats["max_cuda_mem_bytes"] = _max_cuda_mem_bytes()

        diag: Dict[str, Any] = {}
        diag.update(latency_diag)
        diag["valid_host_len_hist"] = dict(valid_len_hist)
        diag["valid_host_len_count"] = int(valid_len_count)
        diag["valid_host_len_sum"] = int(valid_len_sum)
        diag["valid_host_len_mean"] = (float(valid_len_sum) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_min"] = int(valid_len_min) if valid_len_min is not None else None
        diag["valid_host_len_max"] = int(valid_len_max) if valid_len_max is not None else None
        diag["valid_host_len_le_1_frac"] = (float(valid_len_le_1) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_2_frac"] = (float(valid_len_le_2) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_3_frac"] = (float(valid_len_le_3) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_5_frac"] = (float(valid_len_le_5) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_10_frac"] = (float(valid_len_le_10) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["cloud_in_valid_frac"] = (float(cloud_in_valid) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["only_cloud_valid_frac"] = (float(only_cloud_valid) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["no_valid_host_steps"] = int(no_valid_host_steps)
        stats["diagnostics"] = diag
        all_episode_stats.append(stats)

    overall = {
        "eval_episodes": eval_episodes,
        "episodes": all_episode_stats,
        "diagnostics": latency_diag,
    }
    return overall


def evaluate_heft_policy(
        env: Environment,
        cluster_agents: List[EdgeAgent],
        heft_policy: Any,
        eval_episodes: int = 1,
        max_env_steps: int = 300000,
        event_driven: bool = True,
        log_prefix: str = "heft",
) -> Dict[str, Any]:
    eval_episodes = int(eval_episodes)
    max_env_steps = int(max_env_steps)
    inference_target = 1000

    all_episode_stats: List[Dict[str, Any]] = []

    latency_diag = _latency_pair_match_stats(env)

    for ep in range(eval_episodes):
        env.reset()
        if hasattr(heft_policy, "reset"):
            heft_policy.reset()

        _reset_cuda_peak_mem()

        valid_len_hist: Dict[int, int] = defaultdict(int)
        valid_len_count = 0
        valid_len_sum = 0
        valid_len_min: Optional[int] = None
        valid_len_max: Optional[int] = None
        valid_len_le_1 = 0
        valid_len_le_2 = 0
        valid_len_le_3 = 0
        valid_len_le_5 = 0
        valid_len_le_10 = 0
        cloud_ids = {h.id for h in env.all_hosts if getattr(h, "is_cloud_node", False)}
        cloud_in_valid = 0
        only_cloud_valid = 0
        no_valid_host_steps = 0

        stats: Dict[str, Any] = {
            "episode": ep + 1,
            "env_steps": 0,
            "action_steps": 0,
            "success_actions": 0,
            "failed_actions": 0,
            "local_scheduled": 0,
            "remote_scheduled": 0,
            "cloud_scheduled": 0,
            "total_reward": 0.0,
            "no_action_env_steps": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "total_workflows": env.global_stats.get("total_workflows", 0),
            "diagnostics": {},
        }

        env_steps = 0
        no_action_env_steps = 0
        infer_count = 0
        infer_time = 0.0

        while env_steps < max_env_steps:
            all_terminated = True
            for master in env.masters:
                if not master.is_all_workflows_terminated():
                    all_terminated = False
                    break
            if all_terminated:
                break

            executed_any_action = False

            for master_idx, agent in enumerate(cluster_agents):
                if infer_count < inference_target:
                    t0 = time.perf_counter()
                task, chosen_host_id, action_info = heft_policy.select_task_and_host(env, agent, master_idx)
                if infer_count < inference_target:
                    infer_time += float(time.perf_counter() - t0)
                    infer_count += 1

                if task is None or chosen_host_id is None:
                    continue

                valid_hosts = agent.get_valid_hosts_for_task(task)
                valid_host_ids = [h["host_id"] for h in valid_hosts]
                if not valid_host_ids:
                    no_valid_host_steps += 1
                else:
                    vlen = int(len(valid_host_ids))
                    valid_len_hist[vlen] += 1
                    valid_len_count += 1
                    valid_len_sum += vlen
                    if valid_len_min is None or vlen < valid_len_min:
                        valid_len_min = vlen
                    if valid_len_max is None or vlen > valid_len_max:
                        valid_len_max = vlen
                    if vlen <= 1:
                        valid_len_le_1 += 1
                    if vlen <= 2:
                        valid_len_le_2 += 1
                    if vlen <= 3:
                        valid_len_le_3 += 1
                    if vlen <= 5:
                        valid_len_le_5 += 1
                    if vlen <= 10:
                        valid_len_le_10 += 1
                    if cloud_ids and any((cid in valid_host_ids) for cid in cloud_ids):
                        cloud_in_valid += 1
                        if vlen == 1:
                            only_cloud_valid += 1
                success, _ = env.execute_action(master_idx, task.task_id, chosen_host_id)
                reward = agent._calculate_reward(task, success, chosen_host_id, valid_hosts)

                executed_any_action = True
                stats["action_steps"] += 1
                stats["total_reward"] += float(reward)

                if success:
                    stats["success_actions"] += 1
                    host = next(h for h in env.all_hosts if h.id == chosen_host_id)
                    if host in agent.local_hosts:
                        stats["local_scheduled"] += 1
                    elif getattr(host, "is_cloud_node", False):
                        stats["cloud_scheduled"] += 1
                    else:
                        stats["remote_scheduled"] += 1
                else:
                    stats["failed_actions"] += 1

            completed_workflows, failed_workflows = env.step(event_driven=event_driven)
            env_steps += 1
            stats["env_steps"] = env_steps
            stats["completed_workflows"] = int(env.global_stats.get("completed_workflows", 0))
            stats["failed_workflows"] = int(env.global_stats.get("failed_workflows", 0))

            if env_steps % 50 == 0:
                print(
                    f"[eval_{log_prefix}] ep={ep+1}/{eval_episodes} env_steps={env_steps} "
                    f"action_steps={stats['action_steps']} completed={stats['completed_workflows']}/"
                    f"{stats['total_workflows']} failed={stats['failed_workflows']} cur_time={env.cur_time:.2f}",
                    flush=True,
                )

            if not executed_any_action:
                no_action_env_steps += 1
                stats["no_action_env_steps"] = no_action_env_steps
            else:
                no_action_env_steps = 0
                stats["no_action_env_steps"] = 0

        stats["end_time"] = float(env.cur_time)
        stats["global_end_time"] = float(env.cur_time)
        avg_ms, avg_ms_n = _avg_workflow_makespan(env)
        stats["avg_workflow_makespan"] = avg_ms
        stats["avg_workflow_makespan_n"] = int(avg_ms_n)
        stats["inference_decisions_counted"] = int(infer_count)
        stats["inference_1000_decisions_sec"] = float(infer_time) if infer_count >= inference_target else None
        stats["max_rss_kb"] = _max_rss_kb()
        stats["max_cuda_mem_bytes"] = _max_cuda_mem_bytes()

        diag: Dict[str, Any] = {}
        diag.update(latency_diag)
        diag["valid_host_len_hist"] = dict(valid_len_hist)
        diag["valid_host_len_count"] = int(valid_len_count)
        diag["valid_host_len_sum"] = int(valid_len_sum)
        diag["valid_host_len_mean"] = (float(valid_len_sum) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_min"] = int(valid_len_min) if valid_len_min is not None else None
        diag["valid_host_len_max"] = int(valid_len_max) if valid_len_max is not None else None
        diag["valid_host_len_le_1_frac"] = (float(valid_len_le_1) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_2_frac"] = (float(valid_len_le_2) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_3_frac"] = (float(valid_len_le_3) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_5_frac"] = (float(valid_len_le_5) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_10_frac"] = (float(valid_len_le_10) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["cloud_in_valid_frac"] = (float(cloud_in_valid) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["only_cloud_valid_frac"] = (float(only_cloud_valid) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["no_valid_host_steps"] = int(no_valid_host_steps)
        stats["diagnostics"] = diag
        all_episode_stats.append(stats)

    overall = {
        "eval_episodes": eval_episodes,
        "episodes": all_episode_stats,
        "diagnostics": latency_diag,
    }
    return overall


def evaluate_cmmac_policy(
        env: Environment,
        cluster_agents: List[EdgeAgent],
        eval_episodes: int = 1,
        max_env_steps: int = 300000,
        event_driven: bool = True,
) -> Dict[str, Any]:
    eval_episodes = int(eval_episodes)
    max_env_steps = int(max_env_steps)

    inference_target = 1000

    all_episode_stats: List[Dict[str, Any]] = []

    latency_diag = _latency_pair_match_stats(env)

    for ep in range(eval_episodes):
        env.reset()
        for agent in cluster_agents:
            agent.switch_mode(training=False)

        _reset_cuda_peak_mem()

        valid_len_hist: Dict[int, int] = defaultdict(int)
        valid_len_count = 0
        valid_len_sum = 0
        valid_len_min: Optional[int] = None
        valid_len_max: Optional[int] = None
        valid_len_le_1 = 0
        valid_len_le_2 = 0
        valid_len_le_3 = 0
        valid_len_le_5 = 0
        valid_len_le_10 = 0
        cloud_ids = {h.id for h in env.all_hosts if getattr(h, "is_cloud_node", False)}
        cloud_in_valid = 0
        only_cloud_valid = 0
        no_valid_host_steps = 0

        stats: Dict[str, Any] = {
            "episode": ep + 1,
            "env_steps": 0,
            "action_steps": 0,
            "success_actions": 0,
            "failed_actions": 0,
            "local_scheduled": 0,
            "remote_scheduled": 0,
            "cloud_scheduled": 0,
            "total_reward": 0.0,
            "no_action_env_steps": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "total_workflows": env.global_stats.get("total_workflows", 0),
            "diagnostics": {},
        }

        infer_count = 0
        infer_time = 0.0

        env_steps = 0
        no_action_env_steps = 0

        while env_steps < max_env_steps:
            all_terminated = True
            for master in env.masters:
                if not master.is_all_workflows_terminated():
                    all_terminated = False
                    break
            if all_terminated:
                break

            executed_any_action = False

            for master_idx, agent in enumerate(cluster_agents):
                master = env.masters[master_idx]
                agent.update_ready_tasks()

                picked_wf_id = None
                picked_task = None
                for wf_id, tasks in agent.workflow_ready_tasks.items():
                    if not tasks:
                        continue
                    wf = master.workflows.get(wf_id)
                    if not wf or wf.is_completed or wf.is_failed:
                        continue
                    picked_wf_id = wf_id
                    picked_task = tasks[0]
                    break

                if picked_task is None or picked_wf_id is None:
                    continue

                state = env.get_cluster_state(master_idx, focus_task=picked_task)
                valid_hosts = agent.get_valid_hosts_for_task(picked_task)
                valid_host_ids = [h["host_id"] for h in valid_hosts]
                if not valid_host_ids:
                    no_valid_host_steps += 1
                    continue

                vlen = int(len(valid_host_ids))
                valid_len_hist[vlen] += 1
                valid_len_count += 1
                valid_len_sum += vlen
                if valid_len_min is None or vlen < valid_len_min:
                    valid_len_min = vlen
                if valid_len_max is None or vlen > valid_len_max:
                    valid_len_max = vlen
                if vlen <= 1:
                    valid_len_le_1 += 1
                if vlen <= 2:
                    valid_len_le_2 += 1
                if vlen <= 3:
                    valid_len_le_3 += 1
                if vlen <= 5:
                    valid_len_le_5 += 1
                if vlen <= 10:
                    valid_len_le_10 += 1
                if cloud_ids and any((cid in valid_host_ids) for cid in cloud_ids):
                    cloud_in_valid += 1
                    if vlen == 1:
                        only_cloud_valid += 1

                if infer_count < inference_target:
                    _sync_cuda()
                    t0 = time.perf_counter()

                chosen_host_id, _ = agent.algorithm.select_action(
                    state=state,
                    valid_host_ids=valid_host_ids,
                    master_idx=master_idx,
                    epsilon=0.0,
                )

                if infer_count < inference_target:
                    _sync_cuda()
                    infer_time += float(time.perf_counter() - t0)
                    infer_count += 1
                if not chosen_host_id:
                    continue

                success, _ = env.execute_action(master_idx, picked_task.task_id, chosen_host_id)
                reward = agent._calculate_reward(picked_task, success, chosen_host_id, valid_hosts)

                executed_any_action = True
                stats["action_steps"] += 1
                stats["total_reward"] += float(reward)

                if success:
                    stats["success_actions"] += 1
                    host = next(h for h in env.all_hosts if h.id == chosen_host_id)
                    if host in agent.local_hosts:
                        stats["local_scheduled"] += 1
                    elif getattr(host, "is_cloud_node", False):
                        stats["cloud_scheduled"] += 1
                    else:
                        stats["remote_scheduled"] += 1
                else:
                    stats["failed_actions"] += 1

            completed_workflows, failed_workflows = env.step(event_driven=event_driven)
            env_steps += 1
            stats["env_steps"] = env_steps
            stats["completed_workflows"] = int(env.global_stats.get("completed_workflows", 0))
            stats["failed_workflows"] = int(env.global_stats.get("failed_workflows", 0))

            if env_steps % 50 == 0:
                print(
                    f"[eval_cmmac] ep={ep+1}/{eval_episodes} env_steps={env_steps} "
                    f"action_steps={stats['action_steps']} completed={stats['completed_workflows']}/"
                    f"{stats['total_workflows']} failed={stats['failed_workflows']} cur_time={env.cur_time:.2f}",
                    flush=True,
                )

            if not executed_any_action:
                no_action_env_steps += 1
                stats["no_action_env_steps"] = no_action_env_steps
            else:
                no_action_env_steps = 0
                stats["no_action_env_steps"] = 0

        stats["end_time"] = float(env.cur_time)
        stats["global_end_time"] = float(env.cur_time)
        avg_ms, avg_ms_n = _avg_workflow_makespan(env)
        stats["avg_workflow_makespan"] = avg_ms
        stats["avg_workflow_makespan_n"] = int(avg_ms_n)
        stats["inference_decisions_counted"] = int(infer_count)
        stats["inference_1000_decisions_sec"] = float(infer_time) if infer_count >= inference_target else None
        stats["max_rss_kb"] = _max_rss_kb()
        stats["max_cuda_mem_bytes"] = _max_cuda_mem_bytes()

        diag: Dict[str, Any] = {}
        diag.update(latency_diag)
        diag["valid_host_len_hist"] = dict(valid_len_hist)
        diag["valid_host_len_count"] = int(valid_len_count)
        diag["valid_host_len_sum"] = int(valid_len_sum)
        diag["valid_host_len_mean"] = (float(valid_len_sum) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_min"] = int(valid_len_min) if valid_len_min is not None else None
        diag["valid_host_len_max"] = int(valid_len_max) if valid_len_max is not None else None
        diag["valid_host_len_le_1_frac"] = (float(valid_len_le_1) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_2_frac"] = (float(valid_len_le_2) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_3_frac"] = (float(valid_len_le_3) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_5_frac"] = (float(valid_len_le_5) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["valid_host_len_le_10_frac"] = (float(valid_len_le_10) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["cloud_in_valid_frac"] = (float(cloud_in_valid) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["only_cloud_valid_frac"] = (float(only_cloud_valid) / float(valid_len_count)) if valid_len_count > 0 else None
        diag["no_valid_host_steps"] = int(no_valid_host_steps)
        stats["diagnostics"] = diag
        all_episode_stats.append(stats)

    overall = {
        "eval_episodes": eval_episodes,
        "episodes": all_episode_stats,
        "diagnostics": latency_diag,
    }
    return overall

# -------------------------- 环境初始化 --------------------------
def create_env_from_json(cfg: Dict[str, Any], task_csv_paths: List[str]) -> Tuple[Environment, List[Master], List[Node]]:
    """创建云边环境：包含边缘集群（Master）、云节点及工作流"""
    # 1. 云节点（无限资源）
    cloud_node = Node(
        id=cfg["cloud_id"],
        cpu=1e9,
        mem=1e9,
        bw_mbps=10000
    )
    cloud_node.is_cloud_node = True
    # 2. 边缘集群（Master）
    masters = []
    all_hosts = []  # 所有主机（边缘+云）
    for cluster_cfg, csv_path in zip(cfg["edge_clusters"], task_csv_paths):
        edge_hosts = []
        for h in cluster_cfg["hosts"]:
            host = Node(
                id=h["id"],
                cpu=h["cpu_cores"],
                mem=h["ram_mb"] / 1024.0,  # 转换为GB
                bw_mbps=h["bw_mbps"]
            )
            host.is_cloud_node = False
            edge_hosts.append(host)
            all_hosts.append(host)  # 边缘节点先加入
        
        # 初始化Master并加载工作流
        master = Master(
            id=cluster_cfg["id"],
            node_list=edge_hosts,
            latency_to_cloud_ms=cluster_cfg["latency_to_cloud_ms"],
            links=cfg["links"]
        )
        workflows = get_all_task(csv_path)  # 从CSV加载工作流

        for wf in workflows:
            for task in wf.tasks.values():
                task.docker_type = map_task_to_docker_type(task)  # 绑定Docker类型

        master.load_workflows(workflows)
        masters.append(master)

    # 最后添加云节点，确保其在列表末尾
    all_hosts.append(cloud_node)

    # 创建Environment实例
    env = Environment(
        masters=masters,
        all_hosts=all_hosts,
        links=cfg["links"],
        slot_time=1.0
    )
    return env, masters, all_hosts

def deploy_docker_fixed(all_hosts: List[Node], cloud_replicas: int = 2):
    """CPU严格不超限的Docker部署方案"""
    cloud_node = all_hosts[-1]  # 云节点在最后
    edge_hosts = all_hosts[:-1]

    for host in all_hosts:
        host.service_list = []
        host.used_cpu = 0.0
        host.used_mem = 0.0

    docker_types = list(DOCKER_CONFIG.keys())
    docker_types_sorted = sorted(
        docker_types,
        key=lambda dt: (DOCKER_CONFIG[dt]["cpu"], DOCKER_CONFIG[dt]["mem"]),
    )

    def try_add(host: Node, dt: int) -> bool:
        conf = DOCKER_CONFIG[dt]
        if host.used_cpu + conf["cpu"] > host.cpu_max + 1e-9:
            return False
        if host.used_mem + conf["mem"] > host.mem_max + 1e-9:
            return False
        docker = Docker(mem=conf["mem"], cpu=conf["cpu"], kind=dt)
        host.add_docker(docker)
        return True

    for host in edge_hosts:
        if host.cpu_max <= 2.01:
            plan = [(2, 1), (0, 2)]
        else:
            plan = [(2, 2), (0, 2), (3, 1)]

        for dt, cnt in plan:
            if dt not in DOCKER_CONFIG:
                continue
            for _ in range(int(cnt)):
                try_add(host, dt)

        added = True
        while added:
            added = False
            for dt in [0, 2, 3]:
                if dt not in DOCKER_CONFIG:
                    continue
                if try_add(host, dt):
                    added = True

    # 云节点部署所有类型（每类2副本，兜底高并发）
    if cloud_replicas is None or int(cloud_replicas) <= 0:
        cloud_replicas = 0
    cloud_base = max(int(cloud_replicas), 20)
    for dt in DOCKER_CONFIG:
        conf = DOCKER_CONFIG[dt]
        if dt in (0, 2):
            replicas = cloud_base * 3
        elif dt == 3:
            replicas = cloud_base * 2
        else:
            replicas = cloud_base
        for _ in range(int(replicas)):
            docker = Docker(mem=conf["mem"], cpu=conf["cpu"], kind=dt)
            docker.host_id = cloud_node.id
            cloud_node.service_list.append(docker)
    
    print("="*50 + "\nDocker部署完成\n" + "="*50)

def create_cluster_agents(env:Environment) -> Tuple[List[EdgeAgent], CMMACAlgorithmAdapter]:
    """为每个边缘集群创建独立的Agent"""
    cluster_agents = []
    host_ids = [host.id for host in env.all_hosts]
    state_dim = 256

    # 创建全局cmmac算法实例（多集群共享）
    global_algorithm = CMMACAlgorithmAdapter(
        host_ids=host_ids,
        state_dim=state_dim,
        num_masters=len(env.masters),
        gamma=0.95,
        batch_size=32,
        buffer_capacity=100000,
        actor_lr=1e-4,
        critic_lr=1e-3,
        summaries_dir=f"./logs/cmmac_global"
    )
    for master_idx, master in enumerate(env.masters):
        # 创建集群专用的Agent
        agent = EdgeAgent(
            env=env,
            master_id=master_idx,
            algorithm=global_algorithm,
            agent_id=f"cluster_{master_idx}_agent",
            summaries_dir=f"./logs/cluster_{master_idx}",
            training=True,
            actor_lr=1e-4,
            critic_lr=1e-3
        )
        cluster_agents.append(agent)
        print(f"集群 {master_idx} Agent 创建完成")

    return cluster_agents


def create_cluster_agents_ppo(
        env: Environment,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        device: str = "cuda",
) -> List[EdgeAgent]:
    from method.ppo import PPOPolicy, PPOConfig

    cluster_agents: List[EdgeAgent] = []
    host_ids = [host.id for host in env.all_hosts]
    state_dim = 256
    cfg = PPOConfig(device=str(device))
    algo = PPOPolicy(host_ids=host_ids, state_dim=state_dim, cfg=cfg, actor_lr=float(actor_lr), critic_lr=float(critic_lr))

    for master_idx in range(len(env.masters)):
        agent = EdgeAgent(
            env=env,
            master_id=master_idx,
            algorithm=algo,
            agent_id=f"cluster_{master_idx}_ppo",
            summaries_dir=f"./logs/cluster_{master_idx}_ppo",
            training=True,
            actor_lr=float(actor_lr),
            critic_lr=float(critic_lr),
        )
        cluster_agents.append(agent)

    return cluster_agents


def train_ppo_agents(
        env: Environment,
        cluster_agents: List[EdgeAgent],
        episodes: int = 10,
        max_steps: int = 10000,
        early_stop_patience: int = 0,
        early_stop_metric: str = "avg_reward",
        best_save_dir: Optional[str] = None,
        event_driven: bool = False,
        time_penalty_alpha: float = 0.0,
):
    early_stop_patience = int(early_stop_patience or 0)
    early_stop_metric = str(early_stop_metric or "avg_reward")
    if early_stop_metric not in {"avg_reward", "total_reward", "end_time"}:
        raise ValueError(f"unsupported early_stop_metric: {early_stop_metric}")

    best_metric = None
    best_episode = None
    no_improve = 0

    metrics_csv_path: Optional[str] = None
    if best_save_dir:
        metrics_csv_path = os.path.join(os.path.dirname(best_save_dir), "train_metrics.csv")
    metrics_fields = [
        "ts",
        "algo",
        "episode",
        "steps",
        "completed",
        "total",
        "train_end_time",
        "raw_total_reward",
        "base_total_reward",
        "time_penalty_total",
        "total_reward",
        "avg_reward",
        "eval_end_time",
        "metric",
        "best_metric",
        "time_penalty_alpha",
    ]

    max_steps = int(max_steps)
    time_penalty_alpha = float(time_penalty_alpha or 0.0)
    for episode in range(int(episodes)):
        env.reset()
        total_rewards = [0.0 for _ in cluster_agents]
        raw_rewards = [0.0 for _ in cluster_agents]
        base_rewards = [0.0 for _ in cluster_agents]
        episode_time_penalty_total = 0.0
        active_transitions = [None for _ in cluster_agents]
        active_shaped_rewards = [0.0 for _ in cluster_agents]
        steps = 0

        train_end_time = None
        eval_end_time = None
        metric_val = None
        stop_now = False

        while True:
            all_done = True
            all_workflows_terminated = True

            step_start_time = float(env.cur_time)

            for i, agent in enumerate(cluster_agents):
                defer_update = bool(agent.training) and time_penalty_alpha > 0.0
                reward, done = agent.execute_scheduling_step(defer_update=defer_update)
                raw_rewards[i] += float(reward)
                if not defer_update:
                    total_rewards[i] += float(reward)
                    base_rewards[i] += float(reward)
                else:
                    tr_new = getattr(agent, "_pending_transition", None)
                    if tr_new is not None:
                        tr_prev = active_transitions[i]
                        if tr_prev is not None:
                            shaped_reward_prev = float(active_shaped_rewards[i])
                            base_rewards[i] += float(tr_prev["reward"])
                            total_rewards[i] += float(shaped_reward_prev)
                            agent.algorithm.update(
                                state=tr_prev["state"],
                                action=tr_prev["action"],
                                reward=shaped_reward_prev,
                                next_state=tr_prev["next_state"],
                                done=tr_prev["done"],
                                master_idx=agent.master_idx,
                                action_info=tr_prev["action_info"],
                                actor_lr=agent.actor_lr,
                                critic_lr=agent.critic_lr,
                                global_step=agent.global_step,
                            )
                            agent.global_step += 1
                        active_transitions[i] = tr_new
                        active_shaped_rewards[i] = float(tr_new["reward"])
                if not done:
                    all_done = False

                master = env.masters[i]
                if not master.is_all_workflows_terminated():
                    all_workflows_terminated = False

            completed_workflows, failed_workflows = env.step(event_driven=bool(event_driven))
            steps += 1

            if time_penalty_alpha > 0.0:
                delta_t = float(env.cur_time) - step_start_time
                active_indices = [j for j, tr in enumerate(active_transitions) if tr is not None]
                denom = max(1, len(active_indices))
                per_agent_penalty = -time_penalty_alpha * delta_t / float(denom)
                for j in active_indices:
                    active_shaped_rewards[j] += float(per_agent_penalty)
                    episode_time_penalty_total += float(per_agent_penalty)

            if steps % 50 == 0:
                print(
                    f"[Train][ppo] Episode {episode+1}/{episodes} | Step {steps} | "
                    f"Completed: {env.global_stats['completed_workflows']}/"
                    f"{env.global_stats['total_workflows']} | Env time: {env.cur_time:.2f}",
                    flush=True,
                )

            if all_workflows_terminated or steps >= max_steps:
                break

        if time_penalty_alpha > 0.0:
            for i, agent in enumerate(cluster_agents):
                tr_prev = active_transitions[i]
                if tr_prev is None:
                    continue
                shaped_reward_prev = float(active_shaped_rewards[i])
                base_rewards[i] += float(tr_prev["reward"])
                total_rewards[i] += float(shaped_reward_prev)
                agent.algorithm.update(
                    state=tr_prev["state"],
                    action=tr_prev["action"],
                    reward=shaped_reward_prev,
                    next_state=tr_prev["next_state"],
                    done=tr_prev["done"],
                    master_idx=agent.master_idx,
                    action_info=tr_prev["action_info"],
                    actor_lr=agent.actor_lr,
                    critic_lr=agent.critic_lr,
                    global_step=agent.global_step,
                )
                agent.global_step += 1
                active_transitions[i] = None
                active_shaped_rewards[i] = 0.0

        completed = env.global_stats['completed_workflows']
        total = env.global_stats['total_workflows']
        train_end_time = float(env.cur_time)
        raw_total_reward = float(sum(raw_rewards))
        base_total_reward = float(sum(base_rewards))
        time_penalty_total = float(episode_time_penalty_total)
        episode_total_reward = float(sum(total_rewards))
        avg_reward = episode_total_reward / len(cluster_agents) if cluster_agents else 0.0

        print(
            f"[Train][ppo] Episode {episode+1}/{episodes} | "
            f"Completed: {completed}/{total} | "
            f"Steps: {steps} | "
            f"TotalReward: {episode_total_reward:.2f} | "
            f"AvgRewardPerAgent: {avg_reward:.4f}",
            flush=True,
        )

        if early_stop_metric == "end_time":
            eval_results = evaluate_cmmac_policy(
                env,
                cluster_agents,
                eval_episodes=1,
                event_driven=bool(event_driven),
            )
            ep_stats = (eval_results.get("episodes") or [{}])[0]
            eval_end_time = float(ep_stats.get("end_time", float("inf")))
            eval_completed = int(ep_stats.get("completed_workflows", 0))
            eval_total = int(ep_stats.get("total_workflows", 0))
            metric_val = eval_end_time if eval_completed >= eval_total and eval_total > 0 else float("inf")
            for agent in cluster_agents:
                agent.switch_mode(training=True)
        else:
            metric_val = avg_reward if early_stop_metric == "avg_reward" else episode_total_reward

        improved = False
        if best_metric is None:
            improved = True
        else:
            if early_stop_metric == "end_time":
                improved = float(metric_val) < float(best_metric)
            else:
                improved = float(metric_val) > float(best_metric)

        if improved:
            best_metric = float(metric_val)
            best_episode = int(episode) + 1
            no_improve = 0
            if best_save_dir is not None:
                os.makedirs(best_save_dir, exist_ok=True)
                global_algorithm = cluster_agents[0].algorithm if cluster_agents else None
                if global_algorithm is not None and hasattr(global_algorithm, "save_model"):
                    global_algorithm.save_model(best_save_dir)
                    with open(os.path.join(best_save_dir, "meta.json"), "w") as f:
                        json.dump({
                            "best_episode": int(best_episode),
                            "best_metric": float(best_metric),
                            "early_stop_metric": early_stop_metric,
                            "early_stop_patience": int(early_stop_patience),
                            "event_driven": bool(event_driven),
                        }, f, indent=2)
                    print(
                        f"[early_stop][ppo] new best {early_stop_metric}={best_metric:.6f} "
                        f"at episode={best_episode}, saved: {best_save_dir}",
                        flush=True,
                    )
        else:
            if early_stop_patience > 0:
                no_improve += 1
                print(
                    f"[early_stop][ppo] no improve {early_stop_metric}={float(metric_val):.6f} "
                    f"best={float(best_metric):.6f} ({no_improve}/{early_stop_patience})",
                    flush=True,
                )
                if no_improve >= early_stop_patience:
                    print(
                        f"[early_stop][ppo] stop at episode={episode+1} (best_episode={best_episode}, "
                        f"best_{early_stop_metric}={best_metric:.6f})",
                        flush=True,
                    )

                    stop_now = True

        if metrics_csv_path:
            _append_metrics_csv(
                metrics_csv_path,
                metrics_fields,
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "algo": "ppo",
                    "episode": int(episode) + 1,
                    "steps": int(steps),
                    "completed": int(completed),
                    "total": int(total),
                    "train_end_time": float(train_end_time) if train_end_time is not None else "",
                    "raw_total_reward": float(raw_total_reward),
                    "base_total_reward": float(base_total_reward),
                    "time_penalty_total": float(time_penalty_total),
                    "total_reward": float(episode_total_reward),
                    "avg_reward": float(avg_reward),
                    "eval_end_time": float(eval_end_time) if eval_end_time is not None else "",
                    "metric": float(metric_val) if metric_val is not None else "",
                    "best_metric": float(best_metric) if best_metric is not None else "",
                    "time_penalty_alpha": float(time_penalty_alpha),
                },
            )

        if stop_now:
            break

    return cluster_agents

def train_agents(
        env: Environment,
        cluster_agents: List[EdgeAgent],
        episodes: int = 10,
        max_steps: int = 10000,
        early_stop_patience: int = 0,
        early_stop_metric: str = "avg_reward",
        best_save_dir: Optional[str] = None,
        event_driven: bool = False,
        time_penalty_alpha: float = 0.0,
):
    """训练所有集群Agent"""
    print("\n开始训练分布式Actor-Critic模型...")

    early_stop_patience = int(early_stop_patience or 0)
    early_stop_metric = str(early_stop_metric or "avg_reward")
    if early_stop_metric not in {"avg_reward", "total_reward", "end_time"}:
        raise ValueError(f"unsupported early_stop_metric: {early_stop_metric}")

    best_metric = None
    best_episode = None
    no_improve = 0

    metrics_csv_path: Optional[str] = None
    if best_save_dir:
        metrics_csv_path = os.path.join(os.path.dirname(best_save_dir), "train_metrics.csv")
    metrics_fields = [
        "ts",
        "algo",
        "episode",
        "steps",
        "completed",
        "total",
        "train_end_time",
        "raw_total_reward",
        "base_total_reward",
        "time_penalty_total",
        "total_reward",
        "avg_reward",
        "eval_end_time",
        "metric",
        "best_metric",
        "time_penalty_alpha",
    ]

    max_steps = int(max_steps)
    for episode in range(episodes):
        env.reset()
        total_rewards = [0.0 for _ in cluster_agents]
        raw_rewards = [0.0 for _ in cluster_agents]
        base_rewards = [0.0 for _ in cluster_agents]
        episode_time_penalty_total = 0.0
        active_transitions = [None for _ in cluster_agents]
        active_shaped_rewards = [0.0 for _ in cluster_agents]
        steps = 0

        train_end_time = None
        eval_end_time = None
        metric_val = None
        stop_now = False

        while True:
            # 每个集群Agent执行一步调度
            all_done = True
            all_workflows_terminated = True

            step_start_time = float(env.cur_time)

            for i, agent in enumerate(cluster_agents):
                defer_update = bool(agent.training) and time_penalty_alpha > 0.0
                reward, done = agent.execute_scheduling_step(defer_update=defer_update)
                raw_rewards[i] += float(reward)
                if not defer_update:
                    total_rewards[i] += reward
                    base_rewards[i] += float(reward)
                else:
                    tr_new = getattr(agent, "_pending_transition", None)
                    if tr_new is not None:
                        tr_prev = active_transitions[i]
                        if tr_prev is not None:
                            shaped_reward_prev = float(active_shaped_rewards[i])
                            base_rewards[i] += float(tr_prev["reward"])
                            total_rewards[i] += float(shaped_reward_prev)
                            agent.algorithm.update(
                                state=tr_prev["state"],
                                action=tr_prev["action"],
                                reward=shaped_reward_prev,
                                next_state=tr_prev["next_state"],
                                done=tr_prev["done"],
                                master_idx=agent.master_idx,
                                action_info=tr_prev["action_info"],
                                actor_lr=agent.actor_lr,
                                critic_lr=agent.critic_lr,
                                global_step=agent.global_step,
                            )
                            agent.global_step += 1
                        active_transitions[i] = tr_new
                        active_shaped_rewards[i] = float(tr_new["reward"])
                if not done:
                    all_done = False

                # 检查工作流是否全部终止
                master = env.masters[i]
                if not master.is_all_workflows_terminated():
                    all_workflows_terminated = False
            
            # 推进环境时间
            completed_workflows, failed_workflows = env.step(event_driven=bool(event_driven))
            steps += 1

            if time_penalty_alpha > 0.0:
                delta_t = float(env.cur_time) - step_start_time
                active_indices = [j for j, tr in enumerate(active_transitions) if tr is not None]
                denom = max(1, len(active_indices))
                per_agent_penalty = -time_penalty_alpha * delta_t / float(denom)
                for j in active_indices:
                    active_shaped_rewards[j] += float(per_agent_penalty)
                    episode_time_penalty_total += float(per_agent_penalty)

            if steps % 50 == 0:
                print(
                    f"[Train] Episode {episode+1}/{episodes} | Step {steps} | "
                    f"Completed: {env.global_stats['completed_workflows']}/"
                    f"{env.global_stats['total_workflows']} | Env time: {env.cur_time:.2f}"
                )
            
            # 检查终止条件（允许更长的训练序列）
            if all_workflows_terminated or steps >= max_steps:
                break

        if time_penalty_alpha > 0.0:
            for i, agent in enumerate(cluster_agents):
                tr_prev = active_transitions[i]
                if tr_prev is None:
                    continue
                shaped_reward_prev = float(active_shaped_rewards[i])
                base_rewards[i] += float(tr_prev["reward"])
                total_rewards[i] += float(shaped_reward_prev)
                agent.algorithm.update(
                    state=tr_prev["state"],
                    action=tr_prev["action"],
                    reward=shaped_reward_prev,
                    next_state=tr_prev["next_state"],
                    done=tr_prev["done"],
                    master_idx=agent.master_idx,
                    action_info=tr_prev["action_info"],
                    actor_lr=agent.actor_lr,
                    critic_lr=agent.critic_lr,
                    global_step=agent.global_step,
                )
                agent.global_step += 1
                active_transitions[i] = None
                active_shaped_rewards[i] = 0.0

        # 打印训练进度（包括完成工作流数与回合奖励）  
        completed = env.global_stats['completed_workflows']
        total = env.global_stats['total_workflows']
        train_end_time = float(env.cur_time)
        raw_total_reward = float(sum(raw_rewards))
        base_total_reward = float(sum(base_rewards))
        time_penalty_total = float(episode_time_penalty_total)
        episode_total_reward = sum(total_rewards)
        avg_reward = episode_total_reward / len(cluster_agents) if cluster_agents else 0.0

        print(
            f"Episode {episode+1}/{episodes} | "
            f"Completed: {completed}/{total} | "
            f"Steps: {steps} | "
            f"TotalReward: {episode_total_reward:.2f} | "
            f"AvgRewardPerAgent: {avg_reward:.4f}"
        )

        if early_stop_metric == "end_time":
            eval_results = evaluate_cmmac_policy(
                env,
                cluster_agents,
                eval_episodes=1,
                event_driven=bool(event_driven),
            )
            ep_stats = (eval_results.get("episodes") or [{}])[0]
            eval_end_time = float(ep_stats.get("end_time", float("inf")))
            eval_completed = int(ep_stats.get("completed_workflows", 0))
            eval_total = int(ep_stats.get("total_workflows", 0))
            metric_val = eval_end_time if eval_completed >= eval_total and eval_total > 0 else float("inf")
            for agent in cluster_agents:
                agent.switch_mode(training=True)
            print(
                f"[Train][eval] Episode {episode+1}/{episodes} | "
                f"EvalEndTime: {eval_end_time:.4f} | Completed: {eval_completed}/{eval_total}",
                flush=True,
            )
        else:
            metric_val = avg_reward if early_stop_metric == "avg_reward" else episode_total_reward

        improved = False
        if best_metric is None:
            improved = True
        else:
            if early_stop_metric == "end_time":
                improved = float(metric_val) < float(best_metric)
            else:
                improved = float(metric_val) > float(best_metric)

        if improved:
            best_metric = float(metric_val)
            best_episode = int(episode) + 1
            no_improve = 0
            if best_save_dir is not None:
                os.makedirs(best_save_dir, exist_ok=True)
                global_algorithm = cluster_agents[0].algorithm if cluster_agents else None
                if global_algorithm is not None and hasattr(global_algorithm, "save_model"):
                    global_algorithm.save_model(best_save_dir)
                    with open(os.path.join(best_save_dir, "meta.json"), "w") as f:
                        json.dump({
                            "best_episode": int(best_episode),
                            "best_metric": float(best_metric),
                            "early_stop_metric": early_stop_metric,
                            "early_stop_patience": int(early_stop_patience),
                            "event_driven": bool(event_driven),
                        }, f, indent=2)
                    print(
                        f"[early_stop][cmmac] new best {early_stop_metric}={best_metric:.6f} "
                        f"at episode={best_episode}, saved: {best_save_dir}",
                        flush=True,
                    )
        else:
            if early_stop_patience > 0:
                no_improve += 1
                print(
                    f"[early_stop][cmmac] no improve {early_stop_metric}={float(metric_val):.6f} "
                    f"best={float(best_metric):.6f} ({no_improve}/{early_stop_patience})",
                    flush=True,
                )
                if no_improve >= early_stop_patience:
                    print(
                        f"[early_stop][cmmac] stop at episode={episode+1} (best_episode={best_episode}, "
                        f"best_{early_stop_metric}={best_metric:.6f})",
                        flush=True,
                    )

                    stop_now = True

        if metrics_csv_path:
            _append_metrics_csv(
                metrics_csv_path,
                metrics_fields,
                {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "algo": "cmmac",
                    "episode": int(episode) + 1,
                    "steps": int(steps),
                    "completed": int(completed),
                    "total": int(total),
                    "train_end_time": float(train_end_time) if train_end_time is not None else "",
                    "raw_total_reward": float(raw_total_reward),
                    "base_total_reward": float(base_total_reward),
                    "time_penalty_total": float(time_penalty_total),
                    "total_reward": float(episode_total_reward),
                    "avg_reward": float(avg_reward),
                    "eval_end_time": float(eval_end_time) if eval_end_time is not None else "",
                    "metric": float(metric_val) if metric_val is not None else "",
                    "best_metric": float(best_metric) if best_metric is not None else "",
                    "time_penalty_alpha": float(time_penalty_alpha),
                },
            )

        if stop_now:
            break
    
    print("\n训练完成!")
    return cluster_agents

def generate_decision_transformer_dataset(
        env: Environment,
        cluster_agents: List[EdgeAgent],
        save_path: Optional[str] = None,
        converted_output_path: Optional[str] = None,
        max_steps: int = 100000,
        target_action_steps: int = 0,
        num_rollouts: int = 1,
        batch_size: int = 100,
        trajectory_level: str = "workflow",
        collect_epsilon: float = 0.0,
        convert_max_len: int = 0,
        convert_stride: int = 0,
        event_driven: bool = False) -> None:
    """
    生成Decision Transformer格式的数据集
    使用训练好的Actor-Critic模型收集高质量轨迹
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "data", "decision_transformer_dataset")

    if num_rollouts is None or int(num_rollouts) <= 0:
        num_rollouts = 1
    num_rollouts = int(num_rollouts)

    if target_action_steps is None:
        target_action_steps = 0
    target_action_steps = int(target_action_steps)

    for agent in cluster_agents:
        agent.switch_mode(training=False)

    env_steps_total = 0
    action_steps_total = 0

    collector = None
    if trajectory_level == "workflow":
        collector = TrajectoryCollector(max_length=max_steps)

    trajectories = []

    try:
        print("\n开始收集Decision Transformer数据集...")

        rollout_idx = 0
        while rollout_idx < num_rollouts and (target_action_steps <= 0 or action_steps_total < target_action_steps):
            env.reset()
            if collector is None:
                cluster_buffers = [
                    {
                        "workflow_id": f"cluster_{i}_rollout_{rollout_idx}",
                        "states": [],
                        "actions": [],
                        "rewards": [],
                        "dones": [],
                        "timesteps": [],
                        "task_ids": [],
                        "is_failure_steps": [],
                        "failure_reasons": [],
                    }
                    for i in range(len(cluster_agents))
                ]
                cluster_action_steps = [0 for _ in cluster_agents]
            env_steps_in_rollout = 0
            no_action_env_steps = 0
            no_valid_host_docker_counter: Dict[int, int] = defaultdict(int)
            no_action_returned_count = 0

            while env_steps_in_rollout < max_steps and (target_action_steps <= 0 or action_steps_total < target_action_steps):
                all_terminated = True
                for master in env.masters:
                    if not master.is_all_workflows_terminated():
                        all_terminated = False
                        break

                if all_terminated:
                    break

                action_steps_before_env_step = action_steps_total

                for master_idx, agent in enumerate(cluster_agents):
                    master = env.masters[master_idx]

                    agent.update_ready_tasks()
                    for wf_id, tasks in agent.workflow_ready_tasks.items():
                        if not tasks:
                            continue

                        wf = master.workflows.get(wf_id)
                        if not wf or wf.is_completed or wf.is_failed:
                            continue

                        task = tasks[0]

                        state = env.get_cluster_state(master_idx, focus_task=task)
                        valid_hosts = agent.get_valid_hosts_for_task(task)
                        valid_host_ids = [h["host_id"] for h in valid_hosts]

                        if not valid_host_ids:
                            if task.docker_type is None:
                                no_valid_host_docker_counter[-1] += 1
                            else:
                                no_valid_host_docker_counter[int(task.docker_type)] += 1
                            continue

                        chosen_host_id, action_info = agent.algorithm.select_action(
                            state=state,
                            valid_host_ids=valid_host_ids,
                            master_idx=master_idx,
                            epsilon=collect_epsilon
                        )

                        if not chosen_host_id:
                            no_action_returned_count += 1
                            continue

                        action_idx = host_id_to_index(chosen_host_id, env.all_hosts)

                        if collector is not None:
                            wf_key = f"{master_idx}:{wf_id}"
                            collector.add_intermediate(
                                wf_id=wf_key,
                                task_id=task.task_id,
                                state=state,
                                action=action_idx,
                                timestep=action_steps_total
                            )

                        success, reason = env.execute_action(master_idx, task.task_id, chosen_host_id)

                        if success:
                            reward = agent._calculate_reward(task, success, chosen_host_id, valid_hosts)

                            if collector is not None:
                                wf_key = f"{master_idx}:{wf_id}"
                                collector.add_completed_task(
                                    wf_id=wf_key,
                                    task=task,
                                    reward=reward
                                )
                                action_steps_total += 1
                            else:
                                step_idx = cluster_action_steps[master_idx]
                                cluster_buffers[master_idx]["states"].append(np.array(state, dtype=np.float32))
                                cluster_buffers[master_idx]["actions"].append(int(action_idx))
                                cluster_buffers[master_idx]["rewards"].append(float(reward))
                                cluster_buffers[master_idx]["dones"].append(False)
                                cluster_buffers[master_idx]["timesteps"].append(int(step_idx))
                                cluster_buffers[master_idx]["task_ids"].append(task.task_id)
                                cluster_buffers[master_idx]["is_failure_steps"].append(False)
                                cluster_buffers[master_idx]["failure_reasons"].append(None)
                                cluster_action_steps[master_idx] += 1
                                action_steps_total += 1
                        else:
                            task.failure_reason = reason

                            if collector is not None:
                                wf_key = f"{master_idx}:{wf_id}"
                                collector.add_failed_task(
                                    wf_id=wf_key,
                                    task=task,
                                    reward=-5.0
                                )
                                action_steps_total += 1
                            else:
                                step_idx = cluster_action_steps[master_idx]
                                cluster_buffers[master_idx]["states"].append(np.array(state, dtype=np.float32))
                                cluster_buffers[master_idx]["actions"].append(int(action_idx))
                                cluster_buffers[master_idx]["rewards"].append(float(-5.0))
                                cluster_buffers[master_idx]["dones"].append(False)
                                cluster_buffers[master_idx]["timesteps"].append(int(step_idx))
                                cluster_buffers[master_idx]["task_ids"].append(task.task_id)
                                cluster_buffers[master_idx]["is_failure_steps"].append(True)
                                cluster_buffers[master_idx]["failure_reasons"].append(str(reason))
                                cluster_action_steps[master_idx] += 1
                                action_steps_total += 1

                completed_workflows, failed_workflows = env.step(event_driven=bool(event_driven))
                env_steps_in_rollout += 1
                env_steps_total += 1

                if action_steps_total == action_steps_before_env_step:
                    no_action_env_steps += 1
                else:
                    no_action_env_steps = 0
                    no_valid_host_docker_counter.clear()
                    no_action_returned_count = 0

                if no_action_env_steps > max(200, batch_size):
                    print(
                        "\n[DIAG] action_steps stalled "
                        f"for {no_action_env_steps} env steps | "
                        f"rollout={rollout_idx} env_steps_total={env_steps_total} "
                        f"cur_time={env.cur_time:.2f} action_steps={action_steps_total}",
                        flush=True,
                    )
                    if no_action_returned_count > 0:
                        print(f"[DIAG] select_action returned None count={no_action_returned_count}", flush=True)
                    if no_valid_host_docker_counter:
                        top_items = sorted(no_valid_host_docker_counter.items(), key=lambda kv: kv[1], reverse=True)[:10]
                        print(f"[DIAG] no_valid_host docker_type top={top_items}", flush=True)

                        docker_types = [k for k, _ in top_items if k >= 0]
                        if docker_types:
                            for dt in docker_types[:5]:
                                total_cnt = 0
                                free_cnt = 0
                                min_busy_until = None
                                for host in env.all_hosts:
                                    for d in host.service_list:
                                        if int(d.kind) != int(dt):
                                            continue
                                        total_cnt += 1
                                        if d.is_free(env.cur_time):
                                            free_cnt += 1
                                        else:
                                            if d.busy_until is not None:
                                                if min_busy_until is None or float(d.busy_until) < float(min_busy_until):
                                                    min_busy_until = float(d.busy_until)
                                print(
                                    f"[DIAG] docker_type={dt} free/total={free_cnt}/{total_cnt} "
                                    f"min_busy_until={min_busy_until}",
                                    flush=True,
                                )

                    for mi, m in enumerate(env.masters):
                        state_counter = defaultdict(int)
                        for wf in m.workflows.values():
                            for t in wf.tasks.values():
                                state_counter[str(t.state.value)] += 1
                        state_items = sorted(state_counter.items(), key=lambda kv: kv[0])
                        print(f"[DIAG] master={mi} task_states={state_items}", flush=True)

                    no_action_env_steps = 0
                    no_valid_host_docker_counter.clear()
                    no_action_returned_count = 0

                if env_steps_total % batch_size == 0:
                    completed = env.global_stats['completed_workflows']
                    total = env.global_stats['total_workflows']
                    if target_action_steps > 0:
                        target_str = str(target_action_steps)
                    else:
                        target_str = "-"
                    print(
                        f"进度: env_steps={env_steps_total} | action_steps={action_steps_total}/{target_str} | "
                        f"完成工作流: {completed}/{total} | "
                        f"环境时间: {env.cur_time:.2f}s"
                    )

            if collector is None:
                for buf in cluster_buffers:
                    if len(buf["states"]) == 0:
                        continue
                    buf["dones"][-1] = True
                    rewards = np.array(buf["rewards"], dtype=np.float32)
                    buf["returns_to_go"] = rewards[::-1].cumsum()[::-1].astype(np.float32)
                    trajectories.append(
                        {
                            "workflow_id": buf["workflow_id"],
                            "states": np.stack(buf["states"], axis=0).astype(np.float32),
                            "actions": np.array(buf["actions"], dtype=np.int32),
                            "rewards": rewards,
                            "returns_to_go": np.array(buf["returns_to_go"], dtype=np.float32),
                            "dones": np.array(buf["dones"], dtype=bool),
                            "timesteps": np.array(buf["timesteps"], dtype=np.int32),
                            "task_ids": np.array(buf["task_ids"], dtype=str),
                            "is_failure_steps": np.array(buf["is_failure_steps"], dtype=bool),
                            "failure_reasons": np.array([
                                "" if r is None else str(r) for r in buf["failure_reasons"]
                            ], dtype=str),
                        }
                    )

            if collector is not None:
                collector.finalize_all()

            rollout_idx += 1

        os.makedirs(save_path, exist_ok=True)
        collector_path = os.path.join(save_path, "dt_dataset.pkl")

        if collector is not None:
            collector.finalize_all()
            collector.save_all(collector_path)
            completed_count = len(collector.completed_trajectories)
        else:
            with open(collector_path, "wb") as f:
                pickle.dump(trajectories, f)
            completed_count = len(trajectories)

        if converted_output_path is None:
            converted_output_path = os.path.join(
                os.path.dirname(__file__),
                "data",
                "decision_unified",
                "dt_dataset_converted.pkl",
            )
        os.makedirs(os.path.dirname(converted_output_path), exist_ok=True)

        if collector is not None:
            source_trajectories = collector.completed_trajectories
        else:
            source_trajectories = trajectories

        converted = []
        if convert_max_len and int(convert_max_len) > 0:
            stride = int(convert_stride) if convert_stride and int(convert_stride) > 0 else int(convert_max_len)
        else:
            stride = 0

        for traj in source_trajectories:
            states_full = np.asarray(traj["states"], dtype=np.float32)
            actions_full = np.asarray(traj["actions"], dtype=np.float32)
            rewards_full = np.asarray(traj["rewards"], dtype=np.float32)
            dones_full = np.asarray(traj["dones"], dtype=bool)
            n = len(states_full)

            if not (convert_max_len and int(convert_max_len) > 0) or n <= int(convert_max_len):
                slices = [(0, n)]
            else:
                slices = [(s, min(s + int(convert_max_len), n)) for s in range(0, n, stride)]

            base_wf_id = traj.get("workflow_id", "traj")
            for start, end in slices:
                if end - start <= 0:
                    continue
                states = states_full[start:end]
                actions = actions_full[start:end]
                rewards = rewards_full[start:end]
                dones = np.asarray(dones_full[start:end], dtype=bool)
                if len(dones) > 0:
                    dones[-1] = True

                if actions.ndim == 1:
                    actions = actions[:, None]

                converted_traj = {
                    "observations": states,
                    "actions": actions,
                    "rewards": rewards,
                    "dones": dones,
                }

                if "workflow_id" in traj:
                    if start == 0 and end == n:
                        converted_traj["workflow_id"] = traj["workflow_id"]
                    else:
                        converted_traj["workflow_id"] = f"{base_wf_id}_chunk_{start}_{end}"

                if "returns_to_go" in traj:
                    converted_traj["returns_to_go"] = rewards[::-1].cumsum()[::-1].astype(np.float32)

                if "timesteps" in traj:
                    converted_traj["timesteps"] = np.arange(end - start, dtype=np.int32)

                for key in [
                    "task_ids",
                    "is_failure_steps",
                    "failure_reasons",
                ]:
                    if key in traj:
                        converted_traj[key] = np.asarray(traj[key])[start:end]

                converted.append(converted_traj)

        with open(converted_output_path, "wb") as f:
            pickle.dump(converted, f)

        print(f"\n数据集生成完成！")
        print(f"- 保存路径: {save_path}")
        print(f"- 转换后路径: {converted_output_path}")
        traj_label = "完成工作流轨迹数" if collector is not None else "完成集群轨迹数"
        print(f"- {traj_label}: {completed_count}")

    finally:
        for agent in cluster_agents:
            agent.switch_mode(training=True)

def generate_dataset_metadata(trajectories: List[Dict], env: Environment, save_path: str):
    """生成数据集元信息"""
    if not trajectories:
        return
    
    # 统计信息
    total_steps = sum(len(traj['states']) for traj in trajectories)
    failure_steps = sum(np.sum(traj['is_failure_steps']) for traj in trajectories)
    failure_ratio = failure_steps / total_steps if total_steps > 0 else 0.0
    
    metadata = {
        "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_workflows": len(trajectories),
        "total_steps": total_steps,
        "failure_steps": int(failure_steps),
        "failure_ratio": float(failure_ratio),
        "avg_tasks_per_workflow": float(np.mean([len(traj['task_ids']) for traj in trajectories])),
        "state_dimension": len(trajectories[0]['states'][0]),
        "action_space_size": len(env.all_hosts),
        "environment": {
            "num_clusters": len(env.masters),
            "num_hosts": len(env.all_hosts),
            "num_edge_hosts": len([h for h in env.all_hosts if not h.is_cloud_node]),
            "num_cloud_hosts": len([h for h in env.all_hosts if h.is_cloud_node]),
            "slot_time": env.slot_time
        }
    }
    with open(os.path.join(save_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

def execution(
        env_path: str,
        task_csv_paths: List[str],
        episodes: int = 10,
        max_steps: int = 100000,
        trajectory_level: str = "workflow",
        target_action_steps: int = 0,
        num_rollouts: int = 1,
        collect_epsilon: float = 0.25,
        convert_max_len: int = 200,
        convert_stride: int = 50,
        event_driven: bool = True,
        cloud_replicas: int = 2,
        skip_train: bool = False,
        run_collect: bool = True,
        dt_save_path: Optional[str] = None,
        dt_converted_output_path: Optional[str] = None,
        cmmac_model_dir: Optional[str] = None,
        cmmac_save_dir: Optional[str] = None,
        cmmac_early_stop_patience: int = 0,
        cmmac_early_stop_metric: str = "avg_reward",
        time_penalty_alpha: float = 0.0):
    
    """主执行流程"""
    print("="*60, flush=True)
    print("云边环境分布式Actor-Critic训练与DT数据集生成", flush=True)
    print("="*60, flush=True)

    # 1. 加载环境配置
    print("\n1. 加载环境配置...")
    cfg = load_env_from_json(env_path)
    
    # 2. 创建云边环境
    print("\n2. 创建云边环境...")
    env, masters, all_hosts = create_env_from_json(cfg, task_csv_paths)

    # 3. 部署Docker容器
    print("\n3. 部署Docker容器...", flush=True)
    deploy_docker_fixed(all_hosts, cloud_replicas=cloud_replicas)
    
    # 4. 创建集群Agent
    print("\n4. 创建集群Agent...")
    cluster_agents= create_cluster_agents(env)

    if cmmac_model_dir:
        global_algorithm = cluster_agents[0].algorithm if cluster_agents else None
        if global_algorithm is None:
            raise ValueError("no algorithm found for loading")
        global_algorithm.load_model(cmmac_model_dir)
        print(f"[cmmac] expert checkpoint loaded from: {cmmac_model_dir}", flush=True)

    # 5. 训练Agent
    print("\n5. 训练分布式Actor-Critic模型...", flush=True)
    expert_save_dir = None
    expert_init_dir = None
    expert_best_dir = None
    if cmmac_save_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        expert_save_dir = os.path.join(cmmac_save_dir, f"cmmac_expert_{ts}")
        os.makedirs(expert_save_dir, exist_ok=True)
        expert_init_dir = os.path.join(expert_save_dir, "init")
        expert_best_dir = os.path.join(expert_save_dir, "best")

        with open(os.path.join(expert_save_dir, "train_config.json"), "w") as f:
            json.dump({
                "env_path": env_path,
                "task_csv_paths": task_csv_paths,
                "episodes": int(episodes),
                "max_steps": int(max_steps),
                "event_driven": bool(event_driven),
                "cloud_replicas": int(cloud_replicas),
                "cmmac_model_dir": cmmac_model_dir,
                "cmmac_save_dir": cmmac_save_dir,
                "cmmac_early_stop_patience": int(cmmac_early_stop_patience),
                "cmmac_early_stop_metric": str(cmmac_early_stop_metric),
                "time_penalty_alpha": float(time_penalty_alpha),
            }, f, indent=2)

        global_algorithm = cluster_agents[0].algorithm if cluster_agents else None
        if global_algorithm is not None and hasattr(global_algorithm, "save_model"):
            os.makedirs(expert_init_dir, exist_ok=True)
            global_algorithm.save_model(expert_init_dir)
            with open(os.path.join(expert_init_dir, "meta.json"), "w") as f:
                json.dump({
                    "type": "init",
                    "env_path": env_path,
                    "task_csv_paths": task_csv_paths,
                    "episodes_planned": int(episodes),
                }, f, indent=2)
            print(f"[train_cmmac] init checkpoint saved to: {expert_init_dir}", flush=True)

    if not skip_train and int(episodes) > 0:
        cluster_agents = train_agents(
            env,
            cluster_agents,
            episodes=int(episodes),
            max_steps=int(max_steps),
            early_stop_patience=int(cmmac_early_stop_patience),
            early_stop_metric=str(cmmac_early_stop_metric),
            best_save_dir=expert_best_dir,
            event_driven=bool(event_driven),
            time_penalty_alpha=float(time_penalty_alpha),
        )

    if expert_save_dir:
        global_algorithm = cluster_agents[0].algorithm if cluster_agents else None
        if global_algorithm is None:
            raise ValueError("no algorithm found for saving")
        global_algorithm.save_model(expert_save_dir)
        with open(os.path.join(expert_save_dir, "meta.json"), "w") as f:
            json.dump({
                "env_path": env_path,
                "task_csv_paths": task_csv_paths,
                "episodes_planned": int(episodes),
                "cmmac_early_stop_patience": int(cmmac_early_stop_patience),
                "cmmac_early_stop_metric": str(cmmac_early_stop_metric),
                "init_dir": expert_init_dir,
                "best_dir": expert_best_dir,
            }, f, indent=2)
        print(f"[train_cmmac] expert checkpoint saved to: {expert_save_dir}", flush=True)

    if run_collect:
        # 6. 生成Decision Transformer数据集
        print("\n6. 生成Decision Transformer数据集...", flush=True)
        generate_decision_transformer_dataset(
            env,
            cluster_agents,
            save_path=dt_save_path,
            converted_output_path=dt_converted_output_path,
            max_steps=max_steps,
            target_action_steps=target_action_steps,
            num_rollouts=num_rollouts,
            trajectory_level=trajectory_level,
            collect_epsilon=collect_epsilon,
            convert_max_len=convert_max_len,
            convert_stride=convert_stride,
            event_driven=event_driven,
        )

    print("\n所有任务完成！")

if __name__ == "__main__":
    # 3个工作流CSV文件路径（顺序对应3个Master）
    TASK_CSV_PATHS = [
        "/root/autodl-fs/edgecloudschedule/pretrain/datasource/splits/train_collect_10000/master_1.csv",
        "/root/autodl-fs/edgecloudschedule/pretrain/datasource/splits/train_collect_10000/master_2.csv",
        "/root/autodl-fs/edgecloudschedule/pretrain/datasource/splits/train_collect_10000/master_3.csv",
    ]
    # 环境配置文件路径
    ENV_PATH = "/root/autodl-fs/edgecloudschedule/pretrain/datasource/env_cloud_edge1.json"
    # 实验参数配置
    EPISODES = 30
    MAX_STEPS = 300000

    def _str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        v = str(v).strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "train_cmmac",
            "train_ppo",
            "collect_dt",
            "eval_du",
            "eval_cmmac",
            "eval_ppo",
            "eval_heft",
            "eval_est",
            "eval_peft",
        ],
        default="all",
    )
    parser.add_argument("--env_path", type=str, default=ENV_PATH)
    parser.add_argument("--task_csv_paths", nargs="+", default=TASK_CSV_PATHS)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--cloud_replicas", type=int, default=2)
    parser.add_argument("--skip_train", action="store_true")

    parser.add_argument(
        "--trajectory_level",
        choices=["cluster", "workflow"],
        default="workflow",
    )
    parser.add_argument("--target_action_steps", type=int, default=0)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--collect_epsilon", type=float, default=0.25)
    parser.add_argument("--convert_max_len", type=int, default=200)
    parser.add_argument("--convert_stride", type=int, default=50)
    parser.add_argument("--dt_save_path", type=str, default=None)
    parser.add_argument("--dt_converted_output_path", type=str, default=None)
    parser.add_argument(
        "--event_driven",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
    )

    parser.add_argument("--du_model_path", type=str, default=None)
    parser.add_argument(
        "--du_dataset_path",
        type=str,
        default="/root/autodl-fs/edgecloudschedule/pretrain/data/decision_unified/dt_dataset_converted.pkl",
    )
    parser.add_argument("--du_device", type=str, default="cuda")
    parser.add_argument("--du_K", type=int, default=20)
    parser.add_argument("--du_embed_dim", type=int, default=256)
    parser.add_argument("--du_n_layer", type=int, default=3)
    parser.add_argument("--du_n_head", type=int, default=1)
    parser.add_argument("--du_dropout", type=float, default=0.1)
    parser.add_argument("--du_conv_window_size", type=int, default=6)
    parser.add_argument("--du_alpha_T", type=float, default=0.85)
    parser.add_argument("--du_remove_act_embs", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--eval_max_env_steps", type=int, default=300000)

    parser.add_argument("--cmmac_model_dir", type=str, default=None)
    parser.add_argument("--cmmac_save_dir", type=str, default="output/edgecloud_cmmac_expert")

    parser.add_argument("--cmmac_early_stop_patience", type=int, default=0)
    parser.add_argument("--cmmac_early_stop_metric", type=str, default="avg_reward", choices=["avg_reward", "total_reward", "end_time"])

    parser.add_argument("--ppo_model_dir", type=str, default=None)
    parser.add_argument("--ppo_save_dir", type=str, default="output/edgecloud_ppo_expert")
    parser.add_argument("--ppo_device", type=str, default="cuda")
    parser.add_argument("--ppo_actor_lr", type=float, default=3e-4)
    parser.add_argument("--ppo_critic_lr", type=float, default=1e-3)
    parser.add_argument("--ppo_early_stop_patience", type=int, default=0)
    parser.add_argument("--ppo_early_stop_metric", type=str, default="avg_reward", choices=["avg_reward", "total_reward", "end_time"])

    parser.add_argument("--time_penalty_alpha", type=float, default=0.0)

    args = parser.parse_args()

    set_seed(int(args.seed))

    if args.mode == "train_cmmac":
        execution(
            env_path=args.env_path,
            task_csv_paths=args.task_csv_paths,
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            trajectory_level=args.trajectory_level,
            target_action_steps=0,
            num_rollouts=0,
            collect_epsilon=float(args.collect_epsilon),
            convert_max_len=int(args.convert_max_len),
            convert_stride=int(args.convert_stride),
            event_driven=bool(args.event_driven),
            cloud_replicas=int(args.cloud_replicas),
            skip_train=False,
            run_collect=False,
            cmmac_save_dir=args.cmmac_save_dir,
            cmmac_early_stop_patience=int(args.cmmac_early_stop_patience),
            cmmac_early_stop_metric=str(args.cmmac_early_stop_metric),
            time_penalty_alpha=float(args.time_penalty_alpha),
        )
    elif args.mode == "train_ppo":
        cfg = load_env_from_json(args.env_path)
        env, masters, all_hosts = create_env_from_json(cfg, args.task_csv_paths)
        deploy_docker_fixed(all_hosts, cloud_replicas=int(args.cloud_replicas))

        cluster_agents = create_cluster_agents_ppo(
            env,
            actor_lr=float(args.ppo_actor_lr),
            critic_lr=float(args.ppo_critic_lr),
            device=str(args.ppo_device),
        )

        ts = time.strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(str(args.ppo_save_dir), f"ppo_expert_{ts}")
        os.makedirs(save_root, exist_ok=True)
        init_dir = os.path.join(save_root, "init")
        best_dir = os.path.join(save_root, "best")

        algo = cluster_agents[0].algorithm if cluster_agents else None
        if algo is None:
            raise ValueError("no ppo algorithm")
        os.makedirs(init_dir, exist_ok=True)
        algo.save_model(init_dir)
        with open(os.path.join(init_dir, "meta.json"), "w") as f:
            json.dump({
                "type": "init",
                "env_path": args.env_path,
                "task_csv_paths": args.task_csv_paths,
                "episodes_planned": int(args.episodes),
            }, f, indent=2)

        cluster_agents = train_ppo_agents(
            env,
            cluster_agents,
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            early_stop_patience=int(args.ppo_early_stop_patience),
            early_stop_metric=str(args.ppo_early_stop_metric),
            best_save_dir=best_dir,
            event_driven=bool(args.event_driven),
            time_penalty_alpha=float(args.time_penalty_alpha),
        )

        algo = cluster_agents[0].algorithm if cluster_agents else None
        if algo is None:
            raise ValueError("no ppo algorithm for saving")
        algo.save_model(save_root)
        with open(os.path.join(save_root, "meta.json"), "w") as f:
            json.dump({
                "env_path": args.env_path,
                "task_csv_paths": args.task_csv_paths,
                "episodes_planned": int(args.episodes),
                "ppo_early_stop_patience": int(args.ppo_early_stop_patience),
                "ppo_early_stop_metric": str(args.ppo_early_stop_metric),
                "init_dir": init_dir,
                "best_dir": best_dir,
            }, f, indent=2)
        print(f"[train_ppo] expert checkpoint saved to: {save_root}", flush=True)
    elif args.mode == "collect_dt":
        execution(
            env_path=args.env_path,
            task_csv_paths=args.task_csv_paths,
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            trajectory_level=args.trajectory_level,
            target_action_steps=int(args.target_action_steps),
            num_rollouts=int(args.num_rollouts),
            collect_epsilon=float(args.collect_epsilon),
            convert_max_len=int(args.convert_max_len),
            convert_stride=int(args.convert_stride),
            event_driven=bool(args.event_driven),
            cloud_replicas=int(args.cloud_replicas),
            skip_train=True,
            run_collect=True,
            dt_save_path=args.dt_save_path,
            dt_converted_output_path=args.dt_converted_output_path,
            cmmac_model_dir=args.cmmac_model_dir,
        )
    else:
        if args.mode == "eval_du":
            from method.du_policy import DecisionUnifiedPolicy

            cfg = load_env_from_json(args.env_path)
            env, masters, all_hosts = create_env_from_json(cfg, args.task_csv_paths)
            deploy_docker_fixed(all_hosts, cloud_replicas=int(args.cloud_replicas))

            if args.du_model_path is None:
                cand_dir = "/root/autodl-fs/edgecloudschedule/output/edgecloud_du_formal"
                cand_files = []
                if os.path.isdir(cand_dir):
                    for fn in os.listdir(cand_dir):
                        if fn.endswith(".pt"):
                            cand_files.append(os.path.join(cand_dir, fn))
                if not cand_files:
                    raise ValueError("--du_model_path is required (no .pt found under output/edgecloud_du_formal)")
                args.du_model_path = max(cand_files, key=lambda p: os.path.getmtime(p))

            host_ids = [h.id for h in env.all_hosts]
            du_policy = DecisionUnifiedPolicy(
                all_host_ids=host_ids,
                model_path=args.du_model_path,
                dataset_path=args.du_dataset_path,
                device=args.du_device,
                K=int(args.du_K),
                embed_dim=int(args.du_embed_dim),
                n_layer=int(args.du_n_layer),
                n_head=int(args.du_n_head),
                dropout=float(args.du_dropout),
                conv_window_size=int(args.du_conv_window_size),
                alpha_T=float(args.du_alpha_T),
                remove_act_embs=bool(args.du_remove_act_embs),
            )

            cluster_agents = []
            for master_idx in range(len(env.masters)):
                agent = EdgeAgent(
                    env=env,
                    master_id=master_idx,
                    algorithm=du_policy,
                    agent_id=f"cluster_{master_idx}_du",
                    summaries_dir=f"./logs/cluster_{master_idx}_du",
                    training=False,
                )
                cluster_agents.append(agent)

            results = evaluate_du_policy(
                env,
                cluster_agents,
                du_policy,
                eval_episodes=int(args.eval_episodes),
                max_env_steps=int(args.eval_max_env_steps),
                event_driven=bool(args.event_driven),
            )
            results["algo"] = "du"
            results["model_size_bytes"] = _path_size_bytes(args.du_model_path)
            du_n_params, du_w_bytes = _module_param_stats(getattr(du_policy, "model", None))
            results["num_parameters"] = int(du_n_params)
            results["weights_size_bytes"] = int(du_w_bytes)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = "output/edgecloud_du_eval"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"eval_du_{ts}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[eval_du] results saved to: {out_path}")
        elif args.mode == "eval_cmmac":
            cfg = load_env_from_json(args.env_path)
            env, masters, all_hosts = create_env_from_json(cfg, args.task_csv_paths)
            deploy_docker_fixed(all_hosts, cloud_replicas=int(args.cloud_replicas))

            cluster_agents = create_cluster_agents(env)
            if not cluster_agents:
                raise ValueError("no cluster agents created")
            global_algorithm = cluster_agents[0].algorithm
            if args.cmmac_model_dir:
                global_algorithm.load_model(args.cmmac_model_dir)

            results = evaluate_cmmac_policy(
                env,
                cluster_agents,
                eval_episodes=int(args.eval_episodes),
                max_env_steps=int(args.eval_max_env_steps),
                event_driven=bool(args.event_driven),
            )
            results["algo"] = "cmmac"
            results["model_size_bytes"] = _path_size_bytes(args.cmmac_model_dir)
            global_algorithm = cluster_agents[0].algorithm if cluster_agents else None
            if global_algorithm is not None:
                cmmac_modules = [getattr(global_algorithm, "critic", None)]
                dist_agents = getattr(global_algorithm, "distributed_agents", [])
                for da in dist_agents:
                    cmmac_modules.append(getattr(da, "actor", None))
                cmmac_n_params, cmmac_w_bytes = _sum_param_stats(cmmac_modules)
            else:
                cmmac_n_params, cmmac_w_bytes = 0, 0
            results["num_parameters"] = int(cmmac_n_params)
            results["weights_size_bytes"] = int(cmmac_w_bytes)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = "output/edgecloud_cmmac_eval"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"eval_cmmac_{ts}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[eval_cmmac] results saved to: {out_path}")
        elif args.mode == "eval_ppo":
            cfg = load_env_from_json(args.env_path)
            env, masters, all_hosts = create_env_from_json(cfg, args.task_csv_paths)
            deploy_docker_fixed(all_hosts, cloud_replicas=int(args.cloud_replicas))

            cluster_agents = create_cluster_agents_ppo(
                env,
                actor_lr=float(args.ppo_actor_lr),
                critic_lr=float(args.ppo_critic_lr),
                device=str(args.ppo_device),
            )
            for ag in cluster_agents:
                ag.switch_mode(training=False)

            algo = cluster_agents[0].algorithm if cluster_agents else None
            if algo is None:
                raise ValueError("no ppo algorithm")
            if args.ppo_model_dir:
                algo.load_model(args.ppo_model_dir)

            results = evaluate_cmmac_policy(
                env,
                cluster_agents,
                eval_episodes=int(args.eval_episodes),
                max_env_steps=int(args.eval_max_env_steps),
                event_driven=bool(args.event_driven),
            )
            results["algo"] = "ppo"
            results["model_size_bytes"] = _path_size_bytes(args.ppo_model_dir)
            algo = cluster_agents[0].algorithm if cluster_agents else None
            ppo_n_params, ppo_w_bytes = _sum_param_stats([
                getattr(algo, "actor", None),
                getattr(algo, "critic", None),
            ])
            results["num_parameters"] = int(ppo_n_params)
            results["weights_size_bytes"] = int(ppo_w_bytes)

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = "output/edgecloud_ppo_eval"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"eval_ppo_{ts}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[eval_ppo] results saved to: {out_path}")
        elif args.mode == "eval_heft":
            from method.HEFT import HEFTPolicy

            cfg = load_env_from_json(args.env_path)
            env, masters, all_hosts = create_env_from_json(cfg, args.task_csv_paths)
            deploy_docker_fixed(all_hosts, cloud_replicas=int(args.cloud_replicas))

            heft_policy = HEFTPolicy()
            cluster_agents = []
            for master_idx in range(len(env.masters)):
                agent = EdgeAgent(
                    env=env,
                    master_id=master_idx,
                    algorithm=heft_policy,
                    agent_id=f"cluster_{master_idx}_heft",
                    summaries_dir=f"./logs/cluster_{master_idx}_heft",
                    training=False,
                )
                cluster_agents.append(agent)

            results = evaluate_heft_policy(
                env,
                cluster_agents,
                heft_policy,
                eval_episodes=int(args.eval_episodes),
                max_env_steps=int(args.eval_max_env_steps),
                event_driven=bool(args.event_driven),
                log_prefix="heft",
            )
            results["algo"] = "heft"
            results["model_size_bytes"] = 0
            results["num_parameters"] = 0
            results["weights_size_bytes"] = 0

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = "output/edgecloud_heft_eval"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"eval_heft_{ts}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[eval_heft] results saved to: {out_path}")
        elif args.mode == "eval_est":
            from method.EST import ESTPolicy

            cfg = load_env_from_json(args.env_path)
            env, masters, all_hosts = create_env_from_json(cfg, args.task_csv_paths)
            deploy_docker_fixed(all_hosts, cloud_replicas=int(args.cloud_replicas))

            est_policy = ESTPolicy()
            cluster_agents = []
            for master_idx in range(len(env.masters)):
                agent = EdgeAgent(
                    env=env,
                    master_id=master_idx,
                    algorithm=est_policy,
                    agent_id=f"cluster_{master_idx}_est",
                    summaries_dir=f"./logs/cluster_{master_idx}_est",
                    training=False,
                )
                cluster_agents.append(agent)

            results = evaluate_heft_policy(
                env,
                cluster_agents,
                est_policy,
                eval_episodes=int(args.eval_episodes),
                max_env_steps=int(args.eval_max_env_steps),
                event_driven=bool(args.event_driven),
                log_prefix="est",
            )
            results["algo"] = "est"
            results["model_size_bytes"] = 0
            results["num_parameters"] = 0
            results["weights_size_bytes"] = 0

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = "output/edgecloud_est_eval"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"eval_est_{ts}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[eval_est] results saved to: {out_path}")
        elif args.mode == "eval_peft":
            from method.PEFT import PEFTPolicy

            cfg = load_env_from_json(args.env_path)
            env, masters, all_hosts = create_env_from_json(cfg, args.task_csv_paths)
            deploy_docker_fixed(all_hosts, cloud_replicas=int(args.cloud_replicas))

            peft_policy = PEFTPolicy()
            cluster_agents = []
            for master_idx in range(len(env.masters)):
                agent = EdgeAgent(
                    env=env,
                    master_id=master_idx,
                    algorithm=peft_policy,
                    agent_id=f"cluster_{master_idx}_peft",
                    summaries_dir=f"./logs/cluster_{master_idx}_peft",
                    training=False,
                )
                cluster_agents.append(agent)

            results = evaluate_heft_policy(
                env,
                cluster_agents,
                peft_policy,
                eval_episodes=int(args.eval_episodes),
                max_env_steps=int(args.eval_max_env_steps),
                event_driven=bool(args.event_driven),
                log_prefix="peft",
            )
            results["algo"] = "peft"
            results["model_size_bytes"] = 0
            results["num_parameters"] = 0
            results["weights_size_bytes"] = 0

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = "output/edgecloud_peft_eval"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"eval_peft_{ts}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[eval_peft] results saved to: {out_path}")
        else:
            execution(
                env_path=args.env_path,
                task_csv_paths=args.task_csv_paths,
                episodes=int(args.episodes),
                max_steps=int(args.max_steps),
                trajectory_level=args.trajectory_level,
                target_action_steps=int(args.target_action_steps),
                num_rollouts=int(args.num_rollouts),
                collect_epsilon=float(args.collect_epsilon),
                convert_max_len=int(args.convert_max_len),
                convert_stride=int(args.convert_stride),
                event_driven=bool(args.event_driven),
                cloud_replicas=int(args.cloud_replicas),
                skip_train=bool(args.skip_train),
                run_collect=True,
                dt_save_path=args.dt_save_path,
                dt_converted_output_path=args.dt_converted_output_path,
                cmmac_model_dir=args.cmmac_model_dir,
                cmmac_save_dir=(args.cmmac_save_dir if (not bool(args.skip_train) and int(args.episodes) > 0) else None),
                time_penalty_alpha=float(args.time_penalty_alpha),
            )
