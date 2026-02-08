from typing import Any, Dict, List, Optional, Tuple

from method.SchedulingAlgorithm import SchedulingAlgorithm
from data.tasks import Task
from env.environment import Environment
from schedule.edgeagent import EdgeAgent


class ESTPolicy(SchedulingAlgorithm):
    def __init__(self):
        self._last_info: Dict[str, Any] = {}

    def select_task_and_host(
        self,
        env: Environment,
        agent: EdgeAgent,
        master_idx: int,
    ) -> Tuple[Optional[Task], Optional[str], Dict[str, Any]]:
        master = env.masters[int(master_idx)]
        ready_tasks = list(master.pending_tasks)
        if not ready_tasks:
            info = {"reason": "no_ready_tasks"}
            self._last_info = info
            return None, None, info

        best_task: Optional[Task] = None
        best_host: Optional[str] = None
        best_est: Optional[float] = None
        best_delay: Optional[float] = None

        for task in ready_tasks:
            valid_hosts = agent.get_valid_hosts_for_task(task)
            if not valid_hosts:
                continue

            task_best_host = None
            task_best_est = None
            task_best_delay = None

            for h in valid_hosts:
                hid = h.get("host_id")
                delay = float(h.get("transfer_delay", 0.0))
                est = float(env.cur_time) + delay
                if task_best_est is None or est < task_best_est or (est == task_best_est and delay < float(task_best_delay)):
                    task_best_host = hid
                    task_best_est = est
                    task_best_delay = delay

            if task_best_host is None or task_best_est is None:
                continue

            if (
                best_est is None
                or task_best_est < best_est
                or (task_best_est == best_est and float(getattr(task, "duration", 0.0)) < float(getattr(best_task, "duration", 0.0)))
                or (task_best_est == best_est and float(getattr(task, "duration", 0.0)) == float(getattr(best_task, "duration", 0.0)) and str(getattr(task, "task_id", "")) < str(getattr(best_task, "task_id", "")))
            ):
                best_task = task
                best_host = task_best_host
                best_est = task_best_est
                best_delay = task_best_delay

        if best_task is None:
            info = {"reason": "no_valid_hosts_for_any_ready_task"}
            self._last_info = info
            return None, None, info

        info = {
            "task_id": getattr(best_task, "task_id", None),
            "chosen_host_id": best_host,
            "est": best_est,
            "transfer_delay": best_delay,
        }
        self._last_info = info
        return best_task, best_host, info

    def select_action(
        self,
        state: Any,
        valid_host_ids: List[str],
        master_idx: int,
        epsilon: float = 0.0,
    ) -> Tuple[Optional[str], Dict]:
        if not valid_host_ids:
            return None, {"reason": "no_valid_hosts"}
        return valid_host_ids[0], {"reason": "est_select_action_fallback"}

    def update(
        self,
        state: Any,
        action: str,
        reward: float,
        next_state: Any,
        done: bool,
        master_idx: int,
        action_info: Dict,
        actor_lr: float,
        critic_lr: float,
        global_step: int,
    ):
        return None

    def save_model(self, path: str):
        return None

    def load_model(self, path: str):
        return None

    def reset(self):
        self._last_info = {}
