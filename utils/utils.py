# models/utils.py

import random
import numpy as np
import torch
from typing import Any, Dict, Tuple, List, Optional
from collections import defaultdict

from env.nodes import Node, Master, Cloud, Docker
from data.tasks import Task, Workflow 

DOCKER_CONFIG = {
    0: {"cpu": 0.5, "mem": 0.4},   # Docker-01: 0.5核 + 0.4GB
    1: {"cpu": 0.5, "mem": 1.0},   # Docker-02: 0.5核 + 1.0GB
    2: {"cpu": 1.0, "mem": 0.4},   # Docker-03: 1.0核 + 0.4GB
    3: {"cpu": 1.0, "mem": 0.7},   # Docker-04: 1.0核 + 0.7GB
    4: {"cpu": 1.0, "mem": 1.0},   # Docker-05: 1.0核 + 1.0GB
    5: {"cpu": 2.0, "mem": 0.4}    # Docker-06: 2.0核 + 0.4GB
}

NUM_DOCKER_TYPES = len(DOCKER_CONFIG)

class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return self._encode(batch)

    def _encode(self, batch):
        states, actions, rewards, next_states, masks, global_states = zip(*batch)
        return {
            'states':        torch.stack(states),                     # [B, state_dim]
            'actions':       torch.tensor(actions, dtype=torch.long),# [B]
            'rewards':       torch.tensor(rewards, dtype=torch.float),# [B]
            'next_states':   torch.stack(next_states),               # [B, global_state_dim]
            'masks':         torch.stack(masks),                     # [B, action_dim]
            'global_states': torch.stack(global_states)              # [B, global_state_dim]
        }

    def __len__(self):
        return len(self.buffer)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_latency(src_id: str, dst_id: str, links: List[Dict[str, Any]], default_latency_s: float = 0.1) -> float:
    """计算主机间传输延迟（秒）"""
    latency_s, _matched = get_latency_with_match(
        src_id=src_id,
        dst_id=dst_id,
        links=links,
        default_latency_s=default_latency_s,
    )
    return float(latency_s)


def _normalize_link_node_id(node_id: str) -> str:
    if node_id is None:
        return node_id
    if "-h" in node_id:
        return node_id.split("-h", 1)[0]
    return node_id


def _lookup_link_latency_s(src_id: str, dst_id: str, links: List[Dict[str, Any]]) -> Tuple[Optional[float], bool]:
    for link in links:
        s = link.get("src")
        d = link.get("dst")
        if (s == src_id and d == dst_id) or (s == dst_id and d == src_id):
            try:
                return float(link["latency_ms"]) / 1000.0, True
            except Exception:
                return None, False
    return None, False


def get_latency_with_match(
    src_id: str,
    dst_id: str,
    links: List[Dict[str, Any]],
    default_latency_s: float = 0.1,
) -> Tuple[float, bool]:
    if src_id == dst_id:
        return 0.0, True

    latency_s, matched = _lookup_link_latency_s(src_id, dst_id, links)
    if matched and latency_s is not None:
        return float(latency_s), True

    src_norm = _normalize_link_node_id(src_id)
    dst_norm = _normalize_link_node_id(dst_id)
    if src_norm != src_id or dst_norm != dst_id:
        latency_s, matched = _lookup_link_latency_s(src_norm, dst_norm, links)
        if matched and latency_s is not None:
            return float(latency_s), True

    return float(default_latency_s), False

def map_task_to_docker_type(task: Task) -> int:
    """
    当任务的 cpu/mem 不是精确匹配某个docker时，按“最小能满足”的策略选择docker类型：
    - 找到所有 cpu >= task.cpu_req 且 mem >= task.mem_req 的 docker
    - 如果没有，扩展条件为 cpu >= task.cpu_req OR mem >= task.mem_req 的近似选择
    - 返回内存最小的满足项（conservative）或 cpu 最接近的
    将 docker_type 写回 task.docker_type 并返回
    """
    task_cpu = task.cpu_req
    task_mem = task.mem_req
    candidate_dts = []
    for dt, conf in DOCKER_CONFIG.items():
        if conf["cpu"] >= task_cpu and conf["mem"] >= task_mem:
            # 计算浪费（越少越好）
            waste = (conf["cpu"] - task_cpu) + (conf["mem"] - task_mem)
            candidate_dts.append((dt, waste))
    if candidate_dts:
        return min(candidate_dts, key=lambda x: x[1])[0]
    return max(DOCKER_CONFIG.keys(), key=lambda dt: (DOCKER_CONFIG[dt]["cpu"], DOCKER_CONFIG[dt]["mem"]))

def host_id_to_index(host_id: str, all_hosts: List[Node]) -> int:
    """将主机ID转换为索引（用于Trajectory的action存储）"""
    for idx, host in enumerate(all_hosts):
        if host.id == host_id:
            return idx
    return len(all_hosts) - 1  # 默认返回最后一个（云节点）


