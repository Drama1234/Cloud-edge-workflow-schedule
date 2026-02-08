
from typing import Any, Dict, List, Optional, Tuple

from method.SchedulingAlgorithm import SchedulingAlgorithm
from data.tasks import Task
from env.environment import Environment
from schedule.edgeagent import EdgeAgent


class HEFTPolicy(SchedulingAlgorithm):
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

        task = max(
            ready_tasks,
            key=lambda t: (
                float(getattr(t, "cp_remaining", 0.0)),
                float(getattr(t, "duration", 0.0)),
                str(getattr(t, "task_id", "")),
            ),
        )

        valid_hosts = agent.get_valid_hosts_for_task(task)
        if not valid_hosts:
            info = {"reason": "no_valid_hosts", "task_id": getattr(task, "task_id", None)}
            self._last_info = info
            return task, None, info

        best_host = None
        best_eft = None
        best_delay = None
        for h in valid_hosts:
            hid = h.get("host_id")
            delay = float(h.get("transfer_delay", 0.0))
            eft = float(env.cur_time) + delay + float(getattr(task, "duration", 0.0))
            if best_eft is None or eft < best_eft or (eft == best_eft and delay < float(best_delay)):
                best_host = hid
                best_eft = eft
                best_delay = delay

        info = {
            "task_id": getattr(task, "task_id", None),
            "chosen_host_id": best_host,
            "eft": best_eft,
            "transfer_delay": best_delay,
        }
        self._last_info = info
        return task, best_host, info

    def select_action(
        self,
        state: Any,
        valid_host_ids: List[str],
        master_idx: int,
        epsilon: float = 0.0,
    ) -> Tuple[Optional[str], Dict]:
        if not valid_host_ids:
            return None, {"reason": "no_valid_hosts"}
        return valid_host_ids[0], {"reason": "heft_select_action_fallback"}

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
