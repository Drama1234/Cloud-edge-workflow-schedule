from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from data.tasks import Task, Workflow
import time
import uuid
import numpy as np

class Docker:
    """Docker服务类，代表部署的服务实例"""
    def __init__(self, mem: float, cpu: float, kind: int, create_time: float=0.0):
        self.mem = mem            # 内存占用
        self.cpu = cpu            # CPU占用
        self.kind = kind          # 服务类型
        self.create_time = create_time

        self.busy_until = 0.0

        self.task_history: List[dict[str, Any]] = []
        self.host_id: Optional[str] = None
    
    def is_free(self, cur_time: float) -> bool:
        """容器是否可用于新的任务"""
        return cur_time >= (self.busy_until or 0.0)
    
    def assign(self, task: Task, start_time: float) -> None:
        """
        分配任务到 Docker, start_time 为实际执行时间。
        返回 entry_id, 供 complete() 定位。
        """
        # 生成唯一 entry id，便于查找与更新
        entry_id = str(uuid.uuid4())
        # 期望结束时间（基于 task.duration）
        start = float(start_time)
        expected_end = start + float(task.duration)
        self.busy_until = expected_end

        # 标记任务为运行状态
        task.mark_running(start, self.host_id)

        self.task_history.append({
            "entry_id": entry_id,
            "task_id": task.task_id,
            "workflow_id": task.workflow_id,
            "start": start,
            "expected_end": expected_end,
            "actual_end": None,
            "status": "running",
            "docker_type": self.kind,
            "host_id": self.host_id,
            "expected_duration": float(task.duration),
            "actual_duration": None,
            "fail_reason": None,
        })
        return entry_id

    def complete(self, task: Task, entry_id: str, actual_end: float, success: bool = True, fail_reason: str = None):
        """
        根据 entry_id 更新历史条目
        """
        for entry in self.task_history:
            if entry["entry_id"] == entry_id and entry["status"] == "running":
                entry["actual_end"] = actual_end
                entry["actual_duration"] = actual_end - entry["start"]
                entry["status"] = "completed" if success else "failed"
                entry["fail_reason"] = fail_reason
                break
        # 更新任务状态
        if success:
            task.mark_completed(actual_end)
        else:
            task.mark_failed(fail_reason)

        # 释放容器
        if actual_end >= self.busy_until:
            self.busy_until = 0.0

class Node:
    """
    计算节点（edge host 或云上的虚拟 host）
    - task_queue 存放 Task 对象
    - service_list 存放 Docker 实例
    - used_cpu / used_mem 是当前分配出去的资源（可用于粗略的调度约束）
    """
    def __init__(self, id:str, cpu: float, mem: float,bw_mbps: float):
        self.id = id             # 节点ID（可选）
        self.used_cpu = 0.0
        self.used_mem = 0.0
        self.cpu_max = cpu  # 最大CPU资源
        self.mem_max = mem  # 最大内存资源
        self.bw_mbps = bw_mbps  # 节点带宽
        self.is_cloud_node = False

        self.service_list: List[Docker] = []   # Docker 容器
        self.task_queue: List[Task] = []         # 任务对象

    def reset(self) -> None:
        """重置节点状态，但保留节点 id、静态配置和已部署的 Docker"""
        # 清空运行期任务队列
        self.task_queue = []
        # 根据已有 Docker 重新计算资源占用
        self.used_cpu = 0.0
        self.used_mem = 0.0
        for docker in self.service_list:
            self.used_cpu += docker.cpu
            self.used_mem += docker.mem
            # 重置 Docker 运行期状态
            docker.busy_until = 0.0
            docker.task_history.clear()

    def add_docker(self, docker: Docker) -> None:
        """向节点部署容器（被 deploy_docker_fixed 调用）"""
        docker.host_id = self.id  # 绑定关联
        self.service_list.append(docker)
        self.used_cpu += docker.cpu
        self.used_mem += docker.mem
    
    def remove_docker(self, docker: Docker) -> None:
        """移除容器并释放已占用的容器规格"""
        if docker in self.service_list:
            self.service_list.remove(docker)
            self.used_cpu = max(0.0, self.used_cpu - docker.cpu)
            self.used_mem = max(0.0, self.used_mem - docker.mem)
            docker.host_id = None

    def has_free_docker(self, task_docker_type: int, cur_time: float) -> bool:
        """检查是否存在可用的空闲 docker（类型匹配并且空闲）"""
        for d in self.service_list:
            if d.kind == task_docker_type and d.is_free(cur_time):
                return True
        return False

    def get_free_docker(self, task_docker_type: int, cur_time: float):
        """获取一个可执行该 task_docker_type 的空闲容器"""
        for d in self.service_list:
            if d.kind == task_docker_type and d.is_free(cur_time):
                return d
        return None
    
    def add_task(self, task: Task) -> None:
        """把 Task 对象加入本 host 的任务队列"""
        if not isinstance(task, Task):
            raise TypeError("Node.add_task expects a Task object")
        self.task_queue.append(task)
        task.assigned_host = self.id

    def pop_ready_tasks(self) -> List[Task]:
        """返回当前队列中已准备执行（就绪）的任务（不移除）"""
        return [t for t in self.task_queue if t.is_ready and not t.is_scheduled]
    
    def is_cloud(self) -> bool:
        """粗略判断是否为云主机（用于 state 构造）"""
        return self.cpu_max > 1e6
    

class Master:
    """
    Master管理一个edge cluster的一组 Node(hosts)。
    注意:Master 不继承 Node(职责区分:Master 为管理者/agent,Node 为计算实体）。
    """
    def __init__(self, id: str, node_list: List[Node], latency_to_cloud_ms: int, links: List[Dict[str, Any]]):
        self.id: str = id
        self.node_list: List[Node] = node_list              # edge hosts
        self.latency_to_cloud_ms: int = latency_to_cloud_ms
        self.links: List[Dict[str, Any]] = links
        self.workflows:Dict[str, Workflow] = {} # 键为workflow_id

        # RL / 调度运行时辅助结构
        self.pending_tasks: List[Task] = []     # 当前 ready 且未调度的 tasks（Task 对象）
        self.completed_workflows: int = 0
        self.total_workflows: int = 0   # 方便后续计算完成比例
        self.task_cp_cache: Dict[str, Dict[str, float]] = {}  # {wf_id: {task_id: cp_remaining}}
        self.cached_total_tasks: int = 0
        self.cached_completed_tasks: int = 0
        self.cached_active_workflows: int = 0
        self.cached_completed_workflows: int = 0
        self.cached_failed_workflows: int = 0

    def load_workflows(self, wf_list: List[Workflow]):
        """加载 workflows"""
        # 将列表转换为字典（wf.id -> Workflow对象）便于索引
        self.workflows = {wf.id: wf for wf in wf_list}
        self.total_workflows = len(wf_list)
        self.completed_workflows = 0  # 重置计数
        self.cached_total_tasks = int(sum(len(wf.tasks) for wf in self.workflows.values()))
        self.cached_completed_tasks = 0
        self.cached_active_workflows = int(self.total_workflows)
        self.cached_completed_workflows = 0
        self.cached_failed_workflows = 0
        # 预计算每个工作流的任务最长下游路径
        for wf_id, wf in self.workflows.items():
            self.task_cp_cache[wf_id] = self._compute_task_longest_downstream(wf)
        # 初始化pending tasks
        self.refresh_pending_tasks()

    def refresh_pending_tasks(self) -> None:
        """
        更新 pending_tasks 为当前所有 workflow 中的 ready 未调度任务集合。
        （调用者应在每个时间片/事件后触发）
        """
        pending = []
        for wf_id in sorted(self.workflows.keys()):
            wf = self.workflows[wf_id]
            if wf.is_completed or wf.is_failed:
                continue
            # 只选择READY状态且未调度的任务
            task_order = getattr(wf, 'topo_order', None)
            if not task_order:
                task_order = sorted(wf.tasks.keys())
            for task_id in task_order:
                task = wf.tasks[task_id]
                if (task.state == Task.State.READY and 
                    not task.is_scheduled and 
                    not task.is_completed and 
                    not task.is_failed):
                    pending.append(task)
        self.pending_tasks = pending

    def update_completed_workflows(self) -> None:
        """
        检查哪些 workflow 已完成（所有任务完成），更新计数器。
        应在每个调度循环/episode tick 后调用。
        """
        count = 0
        for wf in self.workflows.values():
            if wf.is_completed:
                count += 1
        self.completed_workflows = count
    
    def get_completion_ratio(self) -> float:
        """返回工作流完成比例"""
        if self.total_workflows == 0:
            return 0.0
        return self.completed_workflows / self.total_workflows
    
    def get_task_cp_remaining(self, task: Task) -> float:
        """获取任务的剩余关键路径长度（从缓存读取）"""
        return self.task_cp_cache.get(task.workflow_id, {}).get(task.task_id, task.duration)

    def __repr__(self):
        return (
            f"<Master id={self.id} "
            f"#hosts={len(self.node_list)} "
            f"#wfs={len(self.workflows)} "
            f"completed={self.completed_workflows} "
            f"pending_tasks={len(self.pending_tasks)}>"
        )

    def _compute_task_longest_downstream(self, wf:Workflow) -> Dict[str, float]:
        """
        计算每个 task 到工作流终点的最长剩余时间（包含自身duration）。
        返回 dict: task_id -> longest downstream duration (seconds)
        依赖 wf.topo_order（拓扑从头到尾）存在且正确。
        """
        # 如果 topo_order 为空或 wf 有问题，直接返回 durations
        td = {tid: wf.tasks[tid].duration for tid in wf.tasks}
        try:
            order = wf.topo_order
            # 建立后驱映射：node -> list(successors)
            succs = {tid: [] for tid in order}
            for node, deps in wf.dag.items():
                # deps are parent IDs; for each parent, current node is successor
                for p in deps:
                    if p in succs:
                        succs[p].append(node)
            # DP from sinks backward
            longest = {tid: 0.0 for tid in order}
            for tid in reversed(order):
                dur = float(wf.tasks[tid].duration)
                if not succs.get(tid):
                    longest[tid] = dur
                else:
                    downstream_max = max(longest[s] for s in succs[tid])
                    longest[tid] = dur + downstream_max
            return longest
        except Exception:
            # 保障返回每个task至少有自身duration
            return td
        
    def is_all_workflows_terminated(self) -> bool:
        """
        检查所有工作流是否已终止（完成或失败）
        返回True表示没有活跃的工作流
        """
        for wf in self.workflows.values():
            if not wf.is_completed and not wf.is_failed:
                return False
        return True
    
    def get_active_workflows_count(self) -> int:
        """获取活跃工作流数量（未完成且未失败）"""
        count = 0
        for wf in self.workflows.values():
            if not wf.is_completed and not wf.is_failed:
                count += 1
        return count

    def reset(self) -> None:
        """重置 Master 状态，但保留静态配置"""
        self.pending_tasks.clear()
        self.completed_workflows = 0
        self.total_workflows = len(self.workflows)
        self.task_cp_cache.clear()
        
        # 重置工作流和任务状态
        for wf in self.workflows.values():
            wf.is_completed = False
            wf.is_failed = False
            wf.completed_tasks.clear()
            wf.failed_tasks.clear()
            wf.reward_granted = False
            wf.start_time = None
            wf.end_time = None
            
            for task in wf.tasks.values():
                # 通过状态机重置任务状态，避免直接修改只读属性
                task.state = Task.State.PENDING if task.dependencies else Task.State.READY
                task.failed_attempts = 0
                task.failure_reason = None
                task.waiting_reason = None
                task.assigned_host = None
                task.actual_start_time = None
                task.actual_end_time = None
                task.transfer_cost = 0.0
                task.remaining_deps = len(task.dep_objs)

        # 重置后重新构建 pending_tasks 列表
        self.refresh_pending_tasks()
        self.cached_total_tasks = int(sum(len(wf.tasks) for wf in self.workflows.values()))
        self.cached_completed_tasks = 0
        self.cached_active_workflows = int(self.total_workflows)
        self.cached_completed_workflows = 0
        self.cached_failed_workflows = 0

class Cloud:
    """
    Cloud 是一组主机的容器(通常我们使用一个 super-host)。
    为简单起见,Cloud 不直接继承 Node(但包含 node_list)。
    """
    def __init__(self, node_list: List[Node], links: List[Dict[str, Any]]):
        self.node_list: List[Node] = node_list
        self.links: List[Dict[str, Any]] = links

        # 标记云节点
        for node in self.node_list:
            node.is_cloud_node = True

    def all_service_count(self) -> int:
        return sum(len(h.service_list) for h in self.node_list)


