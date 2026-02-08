import math
import numpy as np
from typing import List, Tuple, Dict, Optional,Any
from collections import defaultdict
from env.nodes import Node, Master, Cloud, Docker
from data.tasks import Task, Workflow
from utils.reason import FailureReason
from utils.utils import *

class Environment:
    """
    环境类：负责维护系统状态并执行动作，支持全局管理和奖励计算。
    重构后特点：
    1. 采用基于状态机的任务管理。
    2. 严格处理依赖关系和数据传输延迟。
    3. 详细记录失败原因。
    4. 在任务完成时，主动触发下游任务的状态更新。
    5. 支持全局工作流奖励计算和统计。
    """
    def __init__(
            self, 
            masters: List[Master], 
            all_hosts: List[Node],
            links: List[Dict[str, Any]],
            slot_time: float = 1.0,
    ):
        self.masters = masters
        self.all_hosts = all_hosts
        self.links = links
        self.slot_time = slot_time

        self.cur_time = 0.0
        self.task_entry_map: Dict[str, Tuple[str, str]] = {}   # {task_id: ()}
        self.delay_matrix, self.host_id_to_idx = self._build_delay_matrix()

        # 全局统计信息
        self.global_stats = {
            'completed_workflows': 0,
            'failed_workflows': 0,
            'total_workflows': sum(len(master.workflows) for master in masters),
            'completed_tasks': 0,
            'failed_tasks': 0,
            'makespan_history': defaultdict(list),  # wf_id -> [makespans]
            'workflow_rewards': defaultdict(float),  # wf_id -> total_reward
        }

    def _build_delay_matrix(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """构建主机间延迟矩阵（去标识化，基于主机索引）"""
        host_ids = [h.id for h in self.all_hosts]
        host_id_to_idx = {hid: i for i, hid in enumerate(host_ids)}
        n_hosts = len(host_ids)
        delay_matrix = np.zeros((n_hosts, n_hosts), dtype=np.float32)
        
        for i in range(n_hosts):
            for j in range(n_hosts):
                delay_matrix[i][j] = get_latency(host_ids[i], host_ids[j], self.links)
        return delay_matrix, host_id_to_idx


    def _compute_ideal_makespan(self, wf: Workflow) -> float:
        """直接从Task的start_time和end_time计算工作流的理想完工时间"""
        if not wf.tasks:
            return 0.0
        
        task_times = []
        for task in wf.tasks.values():
            task_times.append((task.start_time, task.end_time))

        # 理想完工时间 = 最晚结束时间 - 最早开始时间
        min_start = min(start for start, end in task_times)
        max_end = max(end for start, end in task_times)
        ideal_makespan = max_end - min_start
        
        return float(ideal_makespan)
    
    def calculate_workflow_reward(self, wf: Workflow) -> float:
        """
        计算工作流的整体奖励（基于完工时间与理想时间的对比）
        """
        if not wf.is_completed:
            return -5.0  # 未完成惩罚

        # 获取实际完工时间
        makespan = wf.compute_makespan()
        ideal_makespan = self._compute_ideal_makespan(wf)

        reward = 5.0 * (ideal_makespan / makespan)

        # 超时惩罚（如果超过理想时间的2倍）
        if makespan > 2 * ideal_makespan:
            reward -= 3.0

        # 记录奖励和完工时间
        self.global_stats['workflow_rewards'][wf.id] = reward
        self.global_stats['makespan_history'][wf.id].append(makespan)
        
        return reward
    
    def get_cluster_state(self, master_idx: int, focus_task: Optional[Task] = None) -> np.ndarray:
        """
        获取单个集群的状态（Actor使用）
        专注于当前集群的局部信息
        """
        if not (0 <= master_idx < len(self.masters)):
            return np.zeros(256, dtype=np.float32)
        
        master = self.masters[master_idx]
        local_hosts = master.node_list

        # 1. 本地节点特征（只关注当前集群的节点）
        node_feats = []
        for host in sorted(local_hosts, key=lambda h: h.id)[:8]:  # 限制最多8个本地节点
            cpu_util = host.used_cpu / host.cpu_max if host.cpu_max > 0 else 0.0
            mem_util = host.used_mem / host.mem_max if host.mem_max > 0 else 0.0
            queue_len = min(len(host.task_queue) / 10.0, 1.0)
            
            # Docker状态（本地节点的Docker可用性）
            docker_feats = []
            for dt in range(NUM_DOCKER_TYPES):
                free_count = sum(1 for docker in host.service_list if docker.kind == dt and docker.is_free(self.cur_time))
                docker_feats.append(min(free_count / 2.0, 1.0))
            
            node_feats.extend([cpu_util, mem_util, queue_len] + docker_feats)

        # 2. 本地待调度任务特征
        task_feats = []
        pending_tasks = list(master.pending_tasks)
        if focus_task is not None and pending_tasks:
            focus_task_id = getattr(focus_task, 'task_id', None)
            if focus_task_id is not None:
                for i, t in enumerate(pending_tasks):
                    if getattr(t, 'task_id', None) == focus_task_id:
                        pending_tasks.insert(0, pending_tasks.pop(i))
                        break

        pending_tasks = pending_tasks[:5]  # 限制最多5个待调度任务
        for task in pending_tasks:
            # 任务资源需求（归一化）
            cpu_req = min(task.cpu_req / 2.0, 1.0)
            mem_req = min(task.mem_req / 1.0, 1.0)
            duration = min(task.duration / 100.0, 1.0)
            
            # 任务依赖和状态
            dep_count = min(len(task.dependencies) / 5.0, 1.0)
            docker_type = task.docker_type / (NUM_DOCKER_TYPES - 1) if task.docker_type is not None else 0.0
            failed_attempts = task.failed_attempts / task.max_retry if task.max_retry > 0 else 0.0
            
            task_feats.extend([cpu_req, mem_req, duration, dep_count, docker_type, failed_attempts])

        # 3. 集群级统计特征
        total_tasks = int(getattr(master, "cached_total_tasks", 0))
        if total_tasks <= 0:
            total_tasks = int(sum(len(wf.tasks) for wf in master.workflows.values()))

        completed_tasks = int(getattr(master, "cached_completed_tasks", 0))
        if completed_tasks < 0:
            completed_tasks = 0
        task_completion_rate = completed_tasks / (total_tasks + 1e-6)

        active_workflows = int(getattr(master, "cached_active_workflows", 0))
        if active_workflows < 0:
            active_workflows = 0

        completed_workflows = int(getattr(master, "cached_completed_workflows", 0))
        if completed_workflows < 0:
            completed_workflows = 0

        workflow_completion_rate = completed_workflows / (len(master.workflows) + 1e-6)
        
        cluster_feats = [
            len(master.pending_tasks) / 20.0,
            task_completion_rate,
            workflow_completion_rate,
            active_workflows / len(master.workflows) if master.workflows else 0.0,
            self.cur_time / 1000.0
        ]

        # 拼接并标准化
        state = np.concatenate([
            np.array(node_feats),
            np.array(task_feats),
            np.array(cluster_feats)
        ])

        state = np.clip(state, 0, 1)
        if len(state) < 256:
            state = np.pad(state, (0, 256 - len(state)))
        else:
            state = state[:256]
        return state.astype(np.float32)
    
    def get_global_stats(self, cur_time: float) -> Dict[str, Dict[str, float]]:
        """获取所有节点（本地+其他集群+云中心）的资源统计"""
        stats = {}
        for node in self.all_hosts:
            stats[node.id] = {
                "cpu_util":node.used_cpu / node.cpu_max if node.cpu_max > 0 else 0.0,
                "mem_util":node.used_mem / node.mem_max if node.mem_max > 0 else 0.0,
                "queue_len":len(node.task_queue),
                "free_dockers":defaultdict(int),
                "is_local": node in self.masters[0].node_list,  # 是否属于当前Master本地集群
                "is_cloud": node.is_cloud_node
            }
            # 统计各类型空闲容器
            for docker in node.service_list:
                if docker.is_free(cur_time):
                    stats[node.id]["free_dockers"][docker.kind] += 1
        return stats
    
    def execute_action(self, master_idx: int, task_id: str, host_id: str) -> Tuple[bool, Optional[str]]:
        """
        执行调度动作：将任务分配给指定主机。
        """
        if not (0 <= master_idx < len(self.masters)):
            return False, FailureReason.HOST_INDEX_OUT_OF_RANGE.value
        
        master = self.masters[master_idx]

        # 1. 验证任务状态：必须是 READY 且未被调度
        try:
            # 从 master 的 pending_tasks 中查找任务
            task = next(t for t in master.pending_tasks if t.task_id == task_id)
        except StopIteration:
            # 任务不在 READY 队列中
            return False,FailureReason.TASK_NOT_PENDING.value
        
        # 检查任务是否真的就绪（依赖是否满足）
        if not task.is_ready:
            return False, FailureReason.DEPENDENCY_NOT_READY.value
        
        if task.is_scheduled:
            return False, FailureReason.TASK_NOT_PENDING.value

        # 2.验证目标主机
        try:
            target_host = next(h for h in self.all_hosts if h.id == host_id)
        except StopIteration:
            return False, FailureReason.HOST_INDEX_OUT_OF_RANGE.value

        # 3.检查主机是否有匹配且空闲的Docker
        if task.docker_type is None:
            return False, FailureReason.DOCKER_MISMATCH.value
        
        # 检查主机是否支持该Docker类型
        if not any(docker.kind == task.docker_type for docker in target_host.service_list):
            return False, FailureReason.DOCKER_MISMATCH.value

        if not target_host.has_free_docker(task.docker_type, self.cur_time):
            return False, FailureReason.RESOURCE_INSUFFICIENT.value
        
        # 4. 计算传输成本（在获取主机后计算）
        self._compute_transfer_cost(task,target_host)

        # 5. 分配 Docker 并预约任务
        docker = target_host.get_free_docker(task.docker_type, self.cur_time)
        if not docker:
            return False, FailureReason.RESOURCE_INSUFFICIENT.value
        
        # 6.计算任务实际开始执行时间
        actual_start_time = self.cur_time + task.transfer_cost
        entry_id = docker.assign(task, actual_start_time)

        # 7.更新任务状态
        task.assigned_host = host_id
        task.actual_start_time = actual_start_time
        task.actual_end_time = actual_start_time + task.duration
        task.state = Task.State.RUNNING 

        # 8.记录任务调度信息
        self.task_entry_map[task_id] = (host_id, entry_id)

        # 9.将任务添加到目标主机的任务队列
        target_host.add_task(task)

        # 10.任务已调度，不再是 pending 状态
        master.pending_tasks.remove(task)
        
        return True, None
    
    def step(self, event_driven: bool = False) -> Tuple[List[Workflow], List[Workflow]]:
        """
        推进环境一个时间片
        1. 遍历所有主机，处理任务执行完成或失败
        2. 更新工作流状态
        3. 刷新所有 Master 的 pending_tasks 列表
        4. 增加任务失败模拟，完善任务处理逻辑
        return: (本时间片内完成的工作流列表, 本时间片内失败的工作流列表)
        """
        if event_driven:
            next_end_time = None
            for host in self.all_hosts:
                for task in host.task_queue:
                    if task.state == Task.State.RUNNING and not task.is_completed and not task.is_failed:
                        if task.actual_end_time is None:
                            continue
                        if next_end_time is None or float(task.actual_end_time) < float(next_end_time):
                            next_end_time = float(task.actual_end_time)

            if next_end_time is None or next_end_time <= float(self.cur_time):
                self.cur_time += self.slot_time
            else:
                self.cur_time = float(next_end_time)
        else:
            self.cur_time += self.slot_time

        completed_workflows = []
        failed_workflows = []
        # 处理正常任务完成和失败
        tasks_to_complete = []
        tasks_to_fail = []

        # 1. 处理所有主机上的任务
        for host in self.all_hosts:
            for task in host.task_queue:
                if task.state == Task.State.RUNNING and not task.is_completed and not task.is_failed:
                    if self.cur_time >= task.actual_end_time:
                        tasks_to_complete.append((task, host))

        # 1.1 处理完成的任务
        for task, host in tasks_to_complete:
            self._complete_task(task, host)
            

        # 1.2 处理失败的任务
        for task, host, reason in tasks_to_fail:
            self._fail_task(task, host, reason)

        # 2. 检查并更新工作流状态
        completed_wf_ids = set()
        failed_wf_ids = set()

        global_completed_workflows = 0
        global_failed_workflows = 0
        global_completed_tasks = 0

        for master in self.masters:
            completed_cnt = 0
            failed_cnt = 0
            active_cnt = 0
            completed_tasks_cnt = 0

            for wf in master.workflows.values():
                completed_tasks_cnt += int(len(wf.completed_tasks))

                if wf.is_completed:
                    completed_cnt += 1
                    global_completed_workflows += 1
                    if wf.id not in completed_wf_ids:
                        completed_wf_ids.add(wf.id)
                        completed_workflows.append(wf)
                elif wf.is_failed:
                    failed_cnt += 1
                    global_failed_workflows += 1
                    if wf.id not in failed_wf_ids:
                        failed_wf_ids.add(wf.id)
                        failed_workflows.append(wf)
                else:
                    active_cnt += 1

            master.completed_workflows = int(completed_cnt)
            master.cached_completed_workflows = int(completed_cnt)
            master.cached_failed_workflows = int(failed_cnt)
            master.cached_active_workflows = int(active_cnt)
            master.cached_completed_tasks = int(completed_tasks_cnt)

            global_completed_tasks += int(completed_tasks_cnt)

        # 同步全局统计信息，供日志打印使用
        self.global_stats['completed_workflows'] = int(global_completed_workflows)
        self.global_stats['failed_workflows'] = int(global_failed_workflows)
        self.global_stats['completed_tasks'] = int(global_completed_tasks)

        # 3. 刷新所有 Master 的 pending_tasks 列表
        # 这会把所有新变为READY的任务加入调度队列
        for master in self.masters:
            master.refresh_pending_tasks()

        return completed_workflows, failed_workflows
    
    def _compute_transfer_cost(self, task: Task, target_host: Node):
        """计算跨集群依赖的传输延迟"""
        dep_hosts = [t.assigned_host for t in task.dep_objs if t.assigned_host and t.is_completed]
        if not dep_hosts:
            task.transfer_cost = 0.0
            return
        
        max_latency = 0.0
        dst_idx = self.host_id_to_idx[target_host.id]
        for src_host_id in dep_hosts:
            if src_host_id in self.host_id_to_idx:
                src_idx = self.host_id_to_idx[src_host_id]
                max_latency = max(max_latency, self.delay_matrix[src_idx][dst_idx])
        task.transfer_cost = max_latency

    def _complete_task(self, task: Task, host: Node):
        """标记任务为完成，并清理状态"""
        # 从主机队列中移除
        if task in host.task_queue:
            host.task_queue.remove(task)

        # 释放Docker资源
        if task.task_id in self.task_entry_map:
            host_id, entry_id = self.task_entry_map.pop(task.task_id)
            if host_id == host.id: 
                for docker in host.service_list:
                    for entry in docker.task_history:
                        if entry["entry_id"] == entry_id and entry["status"] == "running":
                            docker.complete(task, entry_id, self.cur_time)
                            break
        # 任务完成后，Task.mark_completed 已通过 Workflow.on_task_completed 更新工作流状态

    def _fail_task(self, task: Task, host: Node, reason: str):
        """标记任务为失败，并处理重试逻辑"""
        task.failure_reason = reason

        # 从主机队列中移除
        if task in host.task_queue:
            host.task_queue.remove(task)
        
        # 释放Docker资源
        if task.task_id in self.task_entry_map:
            host_id, entry_id = self.task_entry_map.pop(task.task_id)
            if host_id == host.id:
                for docker in host.service_list:
                    for entry in docker.task_history:
                        if entry["entry_id"] == entry_id and entry["status"] == "running":
                            docker.complete(task, entry_id, self.cur_time, success=False, fail_reason=reason)
                            break
        # Task.mark_failed 已通过 Workflow.on_task_failed 通知工作流，这里只负责重试相关重置
        # 如果未耗尽重试次数，重置状态以便重新调度
        if not task.is_retry_exhausted:
            task.is_failed = False
            task.state = Task.State.WAITING
            task.assigned_host = None
            task.actual_start_time = None
            task.actual_end_time = None
        # 如果任务未耗尽重试次数，它的状态会被重置为 WAITING
        # 在下一次 master.refresh_pending_tasks() 时，
        # 当它的依赖（可能包括它自己的重试）再次满足时，会重新进入 READY 队列

    def _compute_reward(self, completed_tasks: List[Task], completed_workflows: List[Workflow]) -> float:
        """批量计算奖励"""
        total_reward = 0.0

        # 任务完成奖励
        for task in completed_tasks:
            task_cost = (task.duration + task.transfer_cost) / 10.0
            total_reward += max(0.1, 1.0 - task_cost)  # 执行越快奖励越高

        # 工作流完成奖励
        for wf in completed_workflows:
            if not wf.reward_granted:
                makespan = wf.compute_makespan() or 100.0
                wf_reward = math.exp(-makespan / 200.0) * 10.0  # 完成奖励
                total_reward += wf_reward
                wf.reward_granted = True
                
        return total_reward
    
    def reset(self):
        """重置环境到初始状态"""
        self.cur_time = 0.0
        self.task_entry_map.clear()

        # 重置全局统计
        self.global_stats.update({
            'completed_workflows': 0,
            'failed_workflows': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_reward': 0.0,
            'step_count': 0
        })
        self.global_stats['makespan_history'].clear()
        self.global_stats['workflow_rewards'].clear()


        for host in self.all_hosts:
            host.reset()
            
        for master in self.masters:
            master.reset()
    
    


    # # master级状态
    # def get_master_state(self, master_idx: int) -> np.ndarray:
    #     if not (0 <= master_idx < len(self.masters)):
    #         return np.zeros(256, dtype=np.float32)
        
    #     master = self.masters[master_idx]
    #     master_host_ids = [h.id for h in master.node_list]

    #     node_stats = self.get_global_node_stats(self.cur_time)

    #     # 1. 节点特征（资源+位置+网络）
    #     node_feats = []
    #     for node_id, stats in node_stats.items():
    #         # 基础节点特征
    #         base_feats = [
    #             stats['cpu_util'],  # CPU利用率
    #             stats['mem_util'],  # 内存利用率
    #             min(stats["queue_len"] / 20.0, 1.0) # 任务队列长度归一化（假设最大20）
    #         ]
    #         # 各类型Docker空闲数量（覆盖6种类型）
    #         docker_feats = [stats["free_dockers"].get(i, 0)/2.0 for i in range(6)] 

    #         # 节点类型编码(本地=0，其它边缘=1，云=2)
    #         if stats['is_local']:
    #             node_type = 0.0
    #         elif stats['is_cloud']:
    #             node_type = 2.0
    #         else:
    #             node_type = 1.0
    #         # 网络延迟特征
    #         local_host_idx = self.host_id_to_idx[master_host_ids[0]]  # 本地集群任意主机
    #         node_idx = self.host_id_to_idx[node_id]
    #         delay_feat = self.delay_matrix[local_host_idx][node_idx]  
    #         node_feats.extend(base_feats+docker_feats+delay_feat)
        
    #     # 2. 待调度任务特征
    #     task_feats = []
    #     pending_tasks = master.pending_tasks[:10]
    #     for task in pending_tasks:
    #         wf = task.workflow
    #         wf_tasks = wf.tasks.values()
    #         # 计算工作流内的最大值（用于归一化）
    #         max_depth = max(t.depth for t in wf_tasks) + 1e-6
    #         max_duration = max(t.duration for t in wf_tasks) + 1e-6
    #         max_cpu = max(t.cpu_req for t in wf_tasks) + 1e-6
    #         max_mem = max(t.mem_req for t in wf_tasks) + 1e-6
    #         max_deps = max(len(t.dependencies) for t in wf_tasks) + 1e-6
    #         max_cp = max(t.cp_remaining for t in wf_tasks) + 1e-6

    #         # 任务静态特征
    #         task_static = [
    #             task.cpu_req / max_cpu,
    #             task.mem_req / max_mem,
    #             task.duration / max_duration,
    #             task.depth / max_depth,
    #             task.dependencies,
    #             task.docker_type,
    #         ]
    #         # 任务动态特征
    #         task_dynamic = [
    #             task.cp_remaining / max_cp,
    #             1.0 if task.is_ready else 0.0,
    #             task.failed_attempts / task.max_retry if task.max_retry > 0 else 0.0
    #         ]
    #         task_feats.extend(task_static + task_dynamic)
        
    #     # 3.全局工作流特征
    #     wf_stats = defaultdict(int)
    #     for wf in master.workflows.values():
    #         wf_stats['total_wfs'] += 1
    #         wf_stats['completed_wfs'] += 1 if wf.is_completed else 0
    #         wf_stats['failed_wfs'] += 1 if wf.is_failed else 0
        
    #     wf_feats = [
    #         len(master.pending_tasks),
    #         wf_stats['failed_wfs'] / (wf_stats['total_wfs'] + 1e-6),
    #         wf_stats['completed_wfs'] / (wf_stats['total_wfs'] + 1e-6)
    #     ]
    #     # 拼接并标准化
    #     state = np.concatenate([
    #         np.array(node_feats),
    #         np.array(task_feats),
    #         np.array(wf_feats)
    #     ])
    #     state = np.clip(state, 0, 1)
    #     if len(state) < 256:
    #         state = np.pad(state, (0, 256 - len(state)))
    #     else:
    #         state = state[:256]
    #     return state.astype(np.float32)

    