from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from collections import defaultdict
import os, json, datetime
from abc import ABC, abstractmethod

from env.environment import Environment
from data.tasks import Task, Workflow
from env.nodes import Node, Master, Cloud, Docker
from utils.reason import FailureReason
from method.SchedulingAlgorithm import SchedulingAlgorithm 
from utils.utils import *
from trajectories.trajectory import TrajectoryCollector

NUM_DOCKER_TYPES = 6

class EdgeAgent:
    """
    集群智能体（Actor）：专注于分布式Actor-Critic训练
    移除Decision Conformer轨迹收集逻辑，优化训练流程
    """
    def __init__(self,
                 env: Environment,
                 master_id: int,
                 algorithm: SchedulingAlgorithm,
                 agent_id:str = None,
                 summaries_dir: str = "./logs/",
                 training: bool = True,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-3
                 ):
        
        self.env = env
        self.master_idx = master_id
        self.master = env.masters[master_id]
        self.agent_id = agent_id or f"agent_{self.master.id}"
        self.algorithm = algorithm

        # 训练/评估模式
        self.training = training
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # 探索参数
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 10000
        self.global_step = 0

        # 折扣因子（从算法获取或默认）
        self.gamma = getattr(algorithm, 'gamma', 0.99)

        # 按工作流组织的任务队列
        self.workflow_ready_tasks = defaultdict(list)  # wf_id -> [tasks]
        
        # 节点分类
        self.local_hosts = self.master.node_list
        self.all_hosts = env.all_hosts
        self.host_ids = [host.id for host in self.all_hosts]

        self._host_by_id = {h.id: h for h in self.all_hosts}
        self._local_host_ids = {h.id for h in self.local_hosts}
        self._host_type_by_id = {h.id: self._get_host_type(h) for h in self.all_hosts}
        self._docker_kinds_by_host_id = {
            h.id: {getattr(d, "kind", None) for d in getattr(h, "service_list", [])}
            for h in self.all_hosts
        }
        self._inv_cpu_max_by_host_id = {
            h.id: (1.0 / float(h.cpu_max) if float(getattr(h, "cpu_max", 0.0) or 0.0) > 0.0 else 0.0)
            for h in self.all_hosts
        }
        self._inv_mem_max_by_host_id = {
            h.id: (1.0 / float(h.mem_max) if float(getattr(h, "mem_max", 0.0) or 0.0) > 0.0 else 0.0)
            for h in self.all_hosts
        }
        self._host_id_to_idx = getattr(env, "host_id_to_idx", None)
        self._delay_matrix = getattr(env, "delay_matrix", None)

        # 统计信息
        self.task_scheduling_metrics = {
            'success_count': 0,
            'failure_count': 0,
            'local_scheduled': 0,
            'remote_scheduled': 0,
            'cloud_scheduled': 0,
            'reward_history': []
        }
        self.run_stats = defaultdict(list)

        self._pending_transition = None

        # 初始化目录
        self.summaries_dir = os.path.join(summaries_dir, self.agent_id)
        os.makedirs(self.summaries_dir, exist_ok=True)

    @property
    def epsilon(self) -> float:
        """动态计算当前探索率（训练模式下衰减）"""
        if not self.training:
            return 0.0  # 评估模式禁用探索
        decay_factor = np.exp(-self.global_step / self.epsilon_decay)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * decay_factor
    
    def update_ready_tasks(self):
        """更新当前集群的任务队列"""
        # 清空队列
        self.workflow_ready_tasks.clear()
        for wf_id, wf in self.master.workflows.items():
            if wf.is_completed or wf.is_failed:
                continue
            # 按拓扑顺序找第一个就绪任务
            for task_id in wf.topo_order:
                task = wf.tasks[task_id]
                if task.is_ready and not task.is_scheduled:
                    self.workflow_ready_tasks[wf_id].append(task)
                    break

    def _normalize_link_node_id(self, node_id: str) -> str:
        if node_id is None:
            return node_id
        if "-h" in node_id:
            return node_id.split("-h", 1)[0]
        return node_id

    def _lookup_transfer_delay_s(self, src_host_id: str, dst_host_id: str) -> float:
        host_id_to_idx = self._host_id_to_idx
        delay_matrix = self._delay_matrix

        if host_id_to_idx and delay_matrix is not None:
            src_candidates = [src_host_id]
            dst_candidates = [dst_host_id]
            src_norm = self._normalize_link_node_id(src_host_id)
            dst_norm = self._normalize_link_node_id(dst_host_id)
            if src_norm != src_host_id:
                src_candidates.append(src_norm)
            if dst_norm != dst_host_id:
                dst_candidates.append(dst_norm)

            for s in src_candidates:
                si = host_id_to_idx.get(s) if s is not None else None
                if si is None:
                    continue
                for d in dst_candidates:
                    di = host_id_to_idx.get(d) if d is not None else None
                    if di is None:
                        continue
                    try:
                        return float(delay_matrix[si][di])
                    except Exception:
                        continue

        return float(get_latency(src_host_id, dst_host_id, self.env.links))

    def get_valid_hosts_for_task(self, task: Task) -> List[Dict[str, Any]]:
        """筛选任务的有效调度节点（检查Docker兼容性、资源可用性和传输延迟）"""
        valid_hosts = []
        # 获取任务依赖的数据源节点
        dep_host_ids = [t.assigned_host for t in task.dep_objs if t.assigned_host and t.is_completed]
        src_host_id = dep_host_ids[0] if dep_host_ids else self.local_hosts[0].id

        for host in self.all_hosts:
            # 检查Docker类型匹配和可用性
            if task.docker_type is not None:
                # 主机必须支持该 Docker 类型
                kinds = self._docker_kinds_by_host_id.get(host.id)
                if not kinds or task.docker_type not in kinds:
                    continue
                # 且该类型 Docker 目前有空闲副本
                if not host.has_free_docker(task.docker_type, self.env.cur_time):
                    continue

            host_id = host.id
            inv_cpu_max = self._inv_cpu_max_by_host_id.get(host_id, 0.0)
            inv_mem_max = self._inv_mem_max_by_host_id.get(host_id, 0.0)

            valid_hosts.append({
                "host_id": host_id,
                "host_type": self._host_type_by_id.get(host_id) or self._get_host_type(host),
                "transfer_delay": self._lookup_transfer_delay_s(src_host_id, host_id),
                "cpu_util": float(host.used_cpu) * float(inv_cpu_max),
                "mem_util": float(host.used_mem) * float(inv_mem_max),
                "is_local": host_id in self._local_host_ids,
            })   
        return valid_hosts
    
    def _get_host_stats(self, host: Node) -> Dict[str, float]:
        """获取节点资源统计"""
        return {
            "cpu_util": host.used_cpu / host.cpu_max if host.cpu_max > 0 else 0.0,
            "mem_util": host.used_mem / host.mem_max if host.mem_max > 0 else 0.0,
            "queue_len": len(host.task_queue)
        }
    
    def _get_host_type(self, host: Node) -> str:
        """判断节点类型"""
        if host in self.local_hosts:
            return "local"
        elif isinstance(host, Cloud) or getattr(host, 'is_cloud_node', False):
            return "cloud"
        else:
            return "remote"
        
    def select_next_task(self) -> Optional[Tuple[str, Task]]:
        """选择下一个要调度的任务（轮转策略）"""
        self.update_ready_tasks()
        
        for wf_id, tasks in self.workflow_ready_tasks.items():
            if tasks:
                return wf_id, tasks[0]
        return None, None
    
    def _calculate_reward(self, task: Task, success: bool, host_id: str, valid_hosts: List[Dict[str, Any]]) -> float:
        """多维度奖励计算（兼顾调度效率、资源利用率和本地优先策略）"""
        if not success:
            return -2.0  # 失败惩罚

        reward = 1.0

        # 1. 传输延迟奖励：延迟越小，奖励越高
        host_info = next((h for h in valid_hosts if h["host_id"] == host_id), None)

        if host_info:
            # 归一化延迟奖励（0-0.5）
            max_delay = 1.0  # 假设最大可接受延迟1秒
            delay_reward = max(0.0, 0.5 * (1.0 - min(host_info["transfer_delay"] / max_delay, 1.0)))
            reward += delay_reward
        
        # 3. 本地节点优先奖励（鼓励本地化部署）
        if host_info and host_info["host_type"] == "local":
            reward += 0.2

        # 工作流完成奖励
        if task.workflow.is_completed:
            wf_reward = self.env.calculate_workflow_reward(task.workflow)
            reward += wf_reward
        
        return reward
    
    def execute_scheduling_step(self, defer_update: bool = False) -> Tuple[float, bool]:
        """单步调度核心逻辑（Actor-Critic训练核心）"""
        self._pending_transition = None
        # 选择任务
        wf_id, task = self.select_next_task()
        if not task:
            return 0.0, True
        
        # 获取有效节点
        valid_hosts = self.get_valid_hosts_for_task(task)
        if not valid_hosts:
            self.task_scheduling_metrics['failure_count'] += 1
            return -1.0, False  # 无有效节点，调度失败
        
        # 获取状态
        state = self.env.get_cluster_state(self.master_idx, focus_task=task)
        # 获取有效主机ID列表
        valid_host_ids = [h["host_id"] for h in valid_hosts]
        
        # 选择目标节点（与 CMMACAlgorithmAdapter 接口对齐）
        chosen_host_id, action_info = self.algorithm.select_action(
            state=state,
            valid_host_ids=valid_host_ids,
            master_idx=self.master_idx,
            epsilon=self.epsilon
        )
        if not chosen_host_id:
            self.task_scheduling_metrics['failure_count'] += 1
            return -5.0, False  # 动作选择失败

        # 执行调度
        success, reason = self.env.execute_action(self.master_idx, task.task_id, chosen_host_id)

        # 计算奖励
        task_reward = self._calculate_reward(task, success, chosen_host_id, valid_hosts)
        next_state = self.env.get_cluster_state(self.master_idx, focus_task=task)

        # 7.训练模式下更新actor-critic
        if self.training:
            done = task.workflow.is_completed or task.workflow.is_failed
            if bool(defer_update):
                self._pending_transition = {
                    "state": state,
                    "action": chosen_host_id,
                    "reward": float(task_reward),
                    "next_state": next_state,
                    "done": bool(done),
                    "action_info": action_info,
                }
            else:
                # 调用分布式算法更新（包含经验池存储和模型优化）
                self.algorithm.update(
                    state=state,
                    action=chosen_host_id,
                    reward=task_reward,
                    next_state=next_state,
                    done=done,
                    master_idx=self.master_idx,
                    action_info=action_info,
                    actor_lr=self.actor_lr,
                    critic_lr=self.critic_lr,
                    global_step=self.global_step
                )
                self.global_step += 1  # 仅训练模式更新全局步数

        # 8. 更新调度统计
        self._update_scheduling_metrics(success, chosen_host_id, task_reward)
        
        return task_reward, False
    
    def _update_scheduling_metrics(self, success: bool, host_id: str, reward: float):
        """更新调度统计信息（用于训练监控）"""
        if success:
            self.task_scheduling_metrics['success_count'] += 1
            # 统计调度节点类型
            host = self._host_by_id.get(host_id)
            if host is None:
                host = next(h for h in self.all_hosts if h.id == host_id)
            host_type = self._host_type_by_id.get(host_id) or self._get_host_type(host)
            if host_type == "local":
                self.task_scheduling_metrics['local_scheduled'] += 1
            elif host_type == "cloud":
                self.task_scheduling_metrics['cloud_scheduled'] += 1
            else:
                self.task_scheduling_metrics['remote_scheduled'] += 1
        else:
            self.task_scheduling_metrics['failure_count'] += 1
        self.task_scheduling_metrics['reward_history'].append(reward)

    def switch_mode(self, training: bool):
        """切换训练/评估模式（核心参数控制）"""
        self.training = training

    def run_scheduling_cycle(self, max_steps: int = None) -> Dict[str, Any]:
        """完整调度周期（训练/评估的核心执行流程）"""
        # 重置周期统计
        cycle_stats = {
            'total_steps': 0,
            'total_reward': 0.0,
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'local_scheduling_rate': 0.0,
            'makespan_stats': {},
            'start_time': self.env.cur_time,
            'epsilon_history': []
        }
        self.task_scheduling_metrics = defaultdict(int, self.task_scheduling_metrics)  # 重置统计

        step = 0
        while not all(wf.is_completed or wf.is_failed for wf in self.master.workflows.values()):
            if max_steps and step >= max_steps:
                break  # 达到最大步数，强制结束
            
            reward, done = self.execute_scheduling_step()
            cycle_stats['total_steps'] += 1
            cycle_stats['total_reward'] += reward
            cycle_stats['epsilon_history'].append(self.epsilon)
            step += 1
            
            if done:
                break  # 无待调度任务，结束周期

        # 计算周期统计指标
        total_tasks = self.task_scheduling_metrics['success_count'] + self.task_scheduling_metrics['failure_count']
        cycle_stats['success_rate'] = (self.task_scheduling_metrics['success_count'] / total_tasks) if total_tasks > 0 else 0.0
        cycle_stats['avg_reward'] = (cycle_stats['total_reward'] / cycle_stats['total_steps']) if cycle_stats['total_steps'] > 0 else 0.0
        cycle_stats['local_scheduling_rate'] = (self.task_scheduling_metrics['local_scheduled'] / self.task_scheduling_metrics['success_count']) if self.task_scheduling_metrics['success_count'] > 0 else 0.0
        
        # 收集工作流完工时间统计
        for wf_id, wf in self.master.workflows.items():
            if wf.is_completed:
                cycle_stats['makespan_stats'][wf_id] = wf.compute_makespan()
        
        # 补充周期元信息
        cycle_stats['end_time'] = self.env.cur_time
        cycle_stats['total_env_time'] = cycle_stats['end_time'] - cycle_stats['start_time']
        cycle_stats['completed_workflows'] = sum(1 for wf in self.master.workflows.values() if wf.is_completed)
        cycle_stats['total_workflows'] = len(self.master.workflows)
        cycle_stats['avg_epsilon'] = np.mean(cycle_stats['epsilon_history']) if cycle_stats['epsilon_history'] else 0.0

        # 保存周期统计
        self._save_cycle_stats(cycle_stats)
        return cycle_stats
    
    def evaluate(self, num_episodes: int = 1) -> Dict[str, Any]:
        """评估模式：禁用探索，统计稳定调度性能"""
        original_mode = self.training
        self.switch_mode(training=False)  # 切换评估模式（epsilon=0）
        
        eval_results = {
            'episodes': [],
            'overall_stats': {}
        }
        
        for episode in range(num_episodes):
            self.env.reset()  # 重置环境
            episode_stats = self.run_scheduling_cycle()
            episode_stats['episode'] = episode + 1
            eval_results['episodes'].append(episode_stats)
        
        # 计算跨episode平均指标
        if eval_results['episodes']:
            eval_results['overall_stats'] = {
                'avg_success_rate': np.mean([ep['success_rate'] for ep in eval_results['episodes']]),
                'avg_makespan': np.mean([
                    np.mean(list(ep['makespan_stats'].values())) 
                    for ep in eval_results['episodes'] if ep['makespan_stats']
                ]),
                'avg_total_reward': np.mean([ep['total_reward'] for ep in eval_results['episodes']]),
                'avg_local_scheduling_rate': np.mean([ep['local_scheduling_rate'] for ep in eval_results['episodes']]),
                'total_episodes': num_episodes
            }
        
        self.switch_mode(training=original_mode)  # 恢复原始模式
        self._save_eval_results(eval_results)
        return eval_results
    

    


    


        
       
    

    


        

        
    










        



        
    

        
    
    


    


    
    def run_workflow_step(self, episode: int, collector: Optional[TrajectoryCollector] = None, global_timestep: int = 0):
        """执行工作流感知的调度步骤"""
        self.update_workflow_queue()
        master = self.env.masters[self.master_idx]

        total_reward = 0.0
        processed_workflows = []

        # 遍历所有活跃工作流
        for wf_id, task_queue in self.workflow_queues.items():
            if not task_queue:
                continue
                
            wf = master.workflows.get(wf_id)
            if not wf or wf.is_completed or wf.is_failed:
                continue
                
            task = task_queue[0]
            if not task or task.state != Task.State.READY:
                continue
            
            # 获取工作流感知的状态
            state = self.env.get_state(self.master_idx, wf_id=wf_id)
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            # 获取有效主机
            valid_host_ids = self.get_valid_hosts_ids(task)
            
            if not valid_host_ids:
                # 任务失败处理
                reward = -10.0
                total_reward += reward
                
                if collector:
                    collector.add_intermediate(
                        wf_id=wf_id,
                        task_id=task.task_id,
                        state=state,
                        action=-1,
                        timestep=global_timestep
                    )
                    collector.add_failed_task(
                        wf_id=wf_id,
                        task=task,
                        reward=reward
                    )
                continue
            
            # 工作流感知的决策
            epsilon = self._get_epsilon(episode)
            chosen_host_id, action_info = self.algorithm.select_action_for_workflow(
                state=state,
                valid_host_ids=valid_host_ids,
                wf_id=wf_id,
                task_id=task.task_id,
                epsilon=epsilon
            )
            
            action_idx = host_id_to_index(chosen_host_id, self.env.all_hosts)
            
            # 轨迹收集（关联工作流）
            if collector:
                collector.add_intermediate(
                    wf_id=wf_id,
                    task_id=task.task_id,
                    state=state,
                    action=action_idx,
                    timestep=global_timestep
                )
            
            # 执行动作
            success, reason = self.env.execute_action_for_workflow(
                master_idx=self.master_idx,
                wf_id=wf_id,
                task_id=task.task_id,
                host_id=chosen_host_id
            )
            
            # 计算奖励（考虑工作流整体）
            if success:
                reward = self.calculate_workflow_reward(task, wf)
                self.workflow_queues[wf_id].pop(0)  # 移除已完成任务
            else:
                reward = -5.0 * (1 + task.failed_attempts)
            
            total_reward += reward
            
            # 更新轨迹
            if collector:
                if success:
                    collector.add_completed_task(
                        wf_id=wf_id,
                        task=task,
                        reward=reward
                    )
                else:
                    collector.add_failed_task(
                        wf_id=wf_id,
                        task=task,
                        reward=reward
                    )
            
            # 更新算法
            next_state = self.env.get_state(self.master_idx, wf_id=wf_id)
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state, dtype=np.float32)
                
            self.algorithm.update(
                state=state,
                action=action_idx,
                reward=reward,
                next_state=next_state,
                done=wf.is_completed or wf.is_failed,
                master_idx=self.master_idx,
                wf_id=wf_id,  # 传递工作流ID
                action_info=action_info,
                global_step=self.global_step
            )
            
            processed_workflows.append(wf_id)
            self.global_step += 1
        
        return total_reward, processed_workflows
    
    def calculate_workflow_reward(self, task: Task, workflow: Workflow) -> float:
        """计算工作流感知的奖励"""
        reward = 0.0
        
        # 任务完成奖励
        task_exec_time = task.duration + task.transfer_cost
        reward += 1.0 / (task_exec_time + 1e-6) * 0.3
        
        # 工作流进度奖励
        progress = len(workflow.finished_tasks) / len(workflow.tasks)
        reward += progress * 0.2
        
        # 关键路径优化奖励
        if task.cp_remaining > 0:
            reward += (workflow.original_cp_length - workflow.current_cp_length) / workflow.original_cp_length * 0.3
        
        # 工作流完成奖励
        if workflow.is_completed:
            makespan = workflow.compute_makespan()
            reward += np.exp(-makespan / workflow.expected_makespan) * 0.2
        
        return reward


    
    def _safe_execute_action(self, task_id: str, host_id: str) -> Tuple[bool, str]:
        """安全执行动作，防止资源冲突"""
        host = next(h for h in self.env.all_hosts if h.id == host_id)
        task = next(t for t in self.env.masters[self.master_idx].pending_tasks if t.task_id == task_id)
        
        # 二次检查资源状态
        if host.is_cloud_node and not host.has_free_docker(task.docker_type, self.env.cur_time):
            return False, "Cloud resource conflict"
        
        return self.env.execute_action(self.master_idx, task_id, host_id)
    
    def _compute_reward(self, task: Task, success: bool, completed_wfs: List[Workflow], failed_wfs: List[Workflow]) -> float:
        """计算奖励信号"""
        if not success:
            return -1.0
            
        reward = 0.0
        
        # 任务完成奖励
        if task.is_completed:
            execution_time = task.duration + task.transfer_cost
            efficiency_reward = 1.0 / (execution_time + 1e-6)
            reward += efficiency_reward * 0.5
            
        # 工作流完成奖励
        if task.workflow in completed_wfs:
            makespan = task.workflow.compute_makespan()
            makespan_reward = np.exp(-makespan / 200.0) * 2.0
            reward += makespan_reward
            
        # 工作流失败惩罚
        if task.workflow in failed_wfs:
            reward -= 1.0
            
        return reward
    
    def _check_episode_done(self) -> bool:
        """检查回合是否结束"""
        all_wfs = []
        for master in self.env.masters:
            all_wfs.extend(master.workflows.values())
            
        return all(wf.is_completed or wf.is_failed for wf in all_wfs)
    
    def train(self, num_episodes: int, print_interval: int = 10, save_interval: int = 50, model_save_dir: str = "./models/"):
        """训练主循环"""
        os.makedirs(model_save_dir, exist_ok=True)
        
        for episode in range(num_episodes):
            self.env.reset()
            total_rewards = defaultdict(float)
            done = False
            episode_step = 0
            
            while not done:
                rewards, done = self.run_one_step(episode)
                
                for agent_idx, reward in rewards.items():
                    total_rewards[agent_idx] += reward
                episode_step += 1

            # 记录统计
            avg_reward = np.mean(list(total_rewards.values())) if total_rewards else 0
            self.episode_rewards.append(avg_reward)
            self.train_stats['episode_reward'].append(avg_reward)
            self.train_stats['episode_steps'].append(episode_step)
            
            # 打印进度
            if (episode + 1) % print_interval == 0:
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Average Reward: {avg_reward:.2f}, "
                      f"Steps: {episode_step}")
            
            # 保存模型
            if (episode + 1) % save_interval == 0:
                self.save_model(os.path.join(model_save_dir, f"model_episode_{episode + 1}"))
        
        # 保存最终模型和统计
        self.save_model(os.path.join(model_save_dir, "model_final"))
        self._save_training_stats(model_save_dir)
        
        print("\nTraining completed!")
        print(f"Average reward over all episodes: {np.mean(self.episode_rewards):.4f}")

    def evaluate(self, num_eval_episodes: int, model_path: str = None) -> Dict[str, Any]:
        """评估智能体性能"""
        if model_path:
            self.load_model(model_path)
        
        # 关闭探索
        original_epsilon = (self.epsilon_start, self.epsilon_end, self.epsilon_decay)
        self.epsilon_start = self.epsilon_end = 0.0
        eval_results = {
            "per_workflow_metrics": [],
            "episode_summary": [],
            "overall_stats": {}
        }
        for episode in range(num_eval_episodes):
            self.env.reset()
            episode_metrics = self._evaluate_episode()
            eval_results["episode_summary"].append({
                "episode": episode + 1,
                "success_rate": episode_metrics["success_rate"],
                "avg_makespan": episode_metrics["avg_makespan"]
            })
            eval_results["per_workflow_metrics"].extend(episode_metrics["workflow_metrics"])
        # 恢复参数
        self.epsilon_start, self.epsilon_end, self.epsilon_decay = original_epsilon
        # 计算总体统计
        eval_results["overall_stats"] = {
            "total_workflows": len(eval_results["per_workflow_metrics"]),
            "success_rate": np.mean([m["success"] for m in eval_results["per_workflow_metrics"]]),
            "avg_makespan": np.mean([m["makespan"] for m in eval_results["per_workflow_metrics"] if m["success"]])
        }
        # 保存评估结果
        self._save_eval_results(eval_results)
        return eval_results
    
    def _evaluate_episode(self) -> Dict[str, Any]:
        """评估单个episode"""
        episode_metrics = {
            "workflow_metrics": [],
            "success_rate": 0.0
        }
        done = False
        while not done:
            _, done = self.run_one_step(episode=0)

        # 收集结果
        completed_wfs = []
        all_wfs = []

        for master in self.env.masters:
            for wf in master.workflows.values():
                all_wfs.append(wf)
                if wf.is_completed:
                    completed_wfs.append(wf)
                    episode_metrics["workflow_metrics"].append({
                        "workflow_id": wf.id,
                        "success": True,
                        "makespan": wf.compute_makespan(),
                        "tasks_completed": len(wf.finished_tasks),
                        "total_tasks": len(wf.tasks)
                    })
                else:
                    episode_metrics["workflow_metrics"].append({
                        "workflow_id": wf.id,
                        "success": False,
                        "makespan": None,
                        "tasks_completed": len(wf.finished_tasks),
                        "total_tasks": len(wf.tasks)
                    })
        # 计算统计
        if all_wfs:
            episode_metrics["success_rate"] = len(completed_wfs) / len(all_wfs)

        return episode_metrics
    
    def save_model(self, save_path: str):
        """保存模型"""
        if hasattr(self.algorithm, 'save_model'):
            self.algorithm.save_model(save_path)
        
        # 保存训练配置
        config = {
            "state_dim": self.state_dim,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "global_step": self.global_step,
            "num_masters": self.num_masters
        }
        
        with open(os.path.join(save_path, "agent_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_model(self, load_path: str):
        """加载模型"""
        if hasattr(self.algorithm, 'load_model'):
            self.algorithm.load_model(load_path)
        
        # 加载配置
        config_path = os.path.join(load_path, "agent_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            
            self.state_dim = config.get("state_dim", self.state_dim)
            self.gamma = config.get("gamma", self.gamma)
            self.global_step = config.get("global_step", 0)

    def _save_training_stats(self, save_dir: str):
        """保存训练统计"""
        stats = {
            "episode_rewards": self.episode_rewards,
            "train_stats": dict(self.train_stats),
            "global_step": self.global_step,
            "training_completed_at": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, "training_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    
    def _save_eval_results(self, eval_results: Dict[str, Any], save_dir: str = "./"):
        """保存评估结果"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{save_dir}/evaluation_results_{timestamp}.json"
        
        with open(save_path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        print(f"Evaluation results saved to: {save_path}")
    
    def run_scheduling_cycle(self, collector: Optional[TrajectoryCollector] = None) -> Dict[str, Any]:
        """运行完整调度周期"""
        stats = {
            'total_steps': 0,
            'total_reward': 0.0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'local_scheduled': 0,
            'remote_scheduled': 0,
            'cloud_scheduled': 0,
            'start_time': self.env.cur_time,
            'workflow_stats': {}
        }

        global_timestep = 0
        while not all(wf.is_completed or wf.is_failed for wf in self.master.workflows.values()):
            reward, done = self.execute_scheduling_step(collector, global_timestep)
            
            if reward > 0:
                stats['completed_tasks'] += 1
                # 统计调度类型
                last_task_id = next(reversed(self.task_scheduling_info.keys())) if self.task_scheduling_info else None
                if last_task_id:
                    host_id = self.task_scheduling_info[last_task_id]['host_id']
                    host = next(h for h in self.all_hosts if h.id == host_id)
                    if host in self.local_hosts:
                        stats['local_scheduled'] += 1
                    elif isinstance(host, Cloud) or getattr(host, 'is_cloud_node', False):
                        stats['cloud_scheduled'] += 1
                    else:
                        stats['remote_scheduled'] += 1
            elif reward < 0:
                stats['failed_tasks'] += 1
                
            stats['total_reward'] += reward
            stats['total_steps'] += 1
            global_timestep += 1
            
            if done:
                break

        # 收集工作流统计信息
        for wf_id, wf in self.master.workflows.items():
            if wf.is_completed:
                makespan = wf.compute_makespan()
                stats['workflow_stats'][wf_id] = {
                    'makespan': makespan,
                    'status': 'completed',
                    'tasks_completed': len(wf.completed_tasks),
                    'total_tasks': len(wf.tasks),
                    'reward': getattr(wf, 'total_reward', 0.0)
                }
            elif wf.is_failed:
                stats['workflow_stats'][wf_id] = {
                    'status': 'failed',
                    'tasks_completed': len(wf. completed_tasks),
                    'total_tasks': len(wf.tasks)
                }

        stats['end_time'] = self.env.cur_time
        stats['total_time'] = stats['end_time'] - stats['start_time']
        stats['completed_workflows'] = sum(1 for wf in self.master.workflows.values() if wf.is_completed)
        stats['total_workflows'] = len(self.master.workflows)
        stats['success_rate'] = stats['completed_workflows'] / stats['total_workflows'] if stats['total_workflows'] > 0 else 0
        
        self._save_run_stats(stats)
        return stats

    def _save_run_stats(self, stats: Dict[str, Any]):
        """保存运行统计"""
        stats['agent_id'] = self.agent_id
        stats['timestamp'] = datetime.datetime.now().isoformat()
        
        stats_path = os.path.join(self.summaries_dir, f"run_stats_{self.local_step}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)





    

