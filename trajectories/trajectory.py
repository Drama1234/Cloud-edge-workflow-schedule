import numpy as np
import json
import os
import pickle
from typing import List, Dict,Tuple,Any,Optional
from collections import defaultdict
from data.tasks import Task,Workflow
from utils.reason import FailureReason

class Trajectory:
    """单条工作流(workflow)轨迹,符合Decision Transformer输入格式"""
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length

        # 序列数据
        self.states: List[np.ndarray] = []          # 状态序列
        self.actions: List[int] = []         # 动作序列
        self.rewards: List[float] = []              # 奖励序列
        self.returns_to_go: List[float] = []        # 累积回报序列
        self.dones: List[bool] = []                 # 终止标志序列
        self.timesteps: List[int] = []              # 时间步序列
        self.task_ids: List[str] = []               # 关联的任务ID（便于调试和分析）

        # 失败相关标记（便于后续分析和模型理解）
        self.is_failure_steps: List[bool] = []  # 该步是否为失败记录
        self.failure_reasons: List[Optional[str]] = []  # 失败原因

    def add_step(
        self, 
        state: np.ndarray, 
        action: int,  # 修正为int，动作是主机索引
        reward: float, 
        done: bool, 
        timestep: int,
        task_id: str,
        is_failure: bool = False,
        failure_reason: Optional[str] = None
    ):
        """添加一步轨迹数据"""
        if len(self.states) >= self.max_length:
            return
        
        self.states.append(np.array(state))
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.timesteps.append(int(timestep))
        self.task_ids.append(task_id)
        self.is_failure_steps.append(bool(is_failure))
        self.failure_reasons.append(failure_reason)
    
    def compute_returns(self, gamma: float = 0.99):
        """计算累积回报（失败步骤的负奖励会引导模型规避）"""
        self.returns_to_go = []
        future = 0.0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                future = 0.0
            future = r + gamma * future
            self.returns_to_go.insert(0, future)
    
    def to_dict(self, workflow_id: str):
        """转为可序列化格式（用于模型训练）"""
        return {
            "workflow_id": workflow_id,
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "returns_to_go": np.array(self.returns_to_go, dtype=np.float32),
            "dones": np.array(self.dones, dtype=bool),
            "timesteps": np.array(self.timesteps, dtype=np.int32),
            "task_ids": np.array(self.task_ids, dtype=str),
            "is_failure_steps": np.array(self.is_failure_steps, dtype=bool),
            "failure_reasons": np.array(self.failure_reasons, dtype=str)
        }
    
    def save(self, path: str,workflow_id: str):
        """保存轨迹数据到文件"""
        trajectory_dict = self.to_dict(workflow_id=workflow_id)
        # JSON 不支持 ndarray，需要转列表
        serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in trajectory_dict.items()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Trajectory':
        """从文件加载轨迹数据"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"轨迹文件不存在: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        trajectory = cls()
        trajectory.states = [np.array(s) for s in data['states']]
        trajectory.actions = data['actions']
        trajectory.rewards = data['rewards']
        trajectory.returns_to_go = data['returns_to_go']
        trajectory.dones = data['dones']
        trajectory.timesteps = data['timesteps']
        trajectory.task_ids = data['task_ids']
        trajectory.is_failure_steps = data['is_failure_steps']
        trajectory.failure_reasons = data['failure_reasons']
        return trajectory


class TrajectoryCollector:
    """
    工作流级别轨迹收集器：
    - 与 Task.is_completed、Workflow.is_completed 标志位联动
    - 支持任务失败重试、依赖状态变化的轨迹记录
    - 每个 workflow 对应一个 Trajectory
    - workflow 完成后写入全局列表
    - 最终保存为一个 .pkl 供 Decision Transformer 使用
    """
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length
        # workflow_id -> Trajectory
        self.buffers: Dict[str, Trajectory] = {} 
        # 完成的轨迹（完整 episode）
        self.completed_trajectories: List[Dict] = []
        # 中间状态缓存：{wf_id: {task_id: (state, action, timestep)}}
        self.intermediate_cache: Dict[str, Dict[str, Tuple[np.ndarray, int, int]]] = defaultdict(dict)
        # 任务状态跟踪：{wf_id: {task_id: {'completed': bool, 'failed_count': int}}}
        self.task_tracker: Dict[str, Dict[str, Dict[str, Any]]]= defaultdict(
            lambda: defaultdict(lambda: {'completed': False, 'failed_count': 0, 'recorded_failures': 0})
        )
        self.max_failures_per_task = 2    

        # 调试统计：采样阶段接口调用次数
        self._debug_intermediate_calls = 0
        self._debug_completed_calls = 0
        self._debug_failed_calls = 0

    def add_intermediate(
        self,
        wf_id: str,
        task_id: str,
        state: np.ndarray,
        action: int, # 主机索引（与Task.assigned_host一致)
        timestep: int
    ):
        """暂存任务调度时的中间状态（未计算奖励）"""
        self._debug_intermediate_calls += 1
        task_info = self.task_tracker[wf_id][task_id]
        # 失败重试的任务：更新缓存（覆盖旧的调度状态）
        if task_info['failed_count'] > 0 and not task_info['completed']:
            self.intermediate_cache[wf_id][task_id] = (state, action, timestep)
            return
        # 新任务：首次调度时缓存
        if not task_info['completed'] and task_id not in self.intermediate_cache[wf_id]:
            self.intermediate_cache[wf_id][task_id] = (state, action, timestep)

    def add_failed_task(
        self,
        wf_id: str,
        task: Task,
        reward: float = -10.0  # 失败奖励（强负奖励，引导规避）
    ):
        """
        存储有效失败记录（仅保留FailureReason.is_valid_failure的记录）
        """
        self._debug_failed_calls += 1
        task_id = task.task_id
        task_info = self.task_tracker[wf_id][task_id]

        # 过滤条件：有效失败 + 未耗尽重试记录存储次数 + 已调度（有中间状态）
        if (not FailureReason.is_valid_failure(task.failure_reason) 
            or task_info['recorded_failures'] >= self.max_failures_per_task
            or task_id not in self.intermediate_cache[wf_id]):
            return
        
        # 取出调度时的中间状态
        state, action, timestep = self.intermediate_cache[wf_id][task_id]
        # 更新任务跟踪信息
        task_info['failed_count'] += 1
        task_info['recorded_failures'] += 1
        # 判断工作流是否因该任务失败而整体失败
        is_wf_done = task.workflow.is_failed
        
        # 写入失败轨迹（标记is_failure=True）
        self.add_step(
            workflow_id=wf_id,
            state=state,
            action=action,
            reward=reward,
            done=is_wf_done,
            timestep=timestep,
            task_id=task_id,
            is_failure=True,
            failure_reason=task.failure_reason
        )

    def add_completed_task(
        self,
        wf_id: str,
        task: Task,
        reward: float,
    ):
        """存储成功记录（原逻辑不变，补充失败标记）"""
        self._debug_completed_calls += 1
        task_id = task.task_id
        task_info = self.task_tracker[wf_id][task_id]

        if task_info['completed'] or task_id not in self.intermediate_cache[wf_id]:
            return
        state, action, timestep = self.intermediate_cache[wf_id].pop(task_id)
        task_info['completed'] = True
        is_wf_done = task.workflow.is_completed
        # 写入成功轨迹（is_failure=False）
        self.add_step(
            workflow_id=wf_id,
            state=state,
            action=action,
            reward=reward,
            done=is_wf_done,
            timestep=timestep,
            task_id=task_id,
            is_failure=False,
            failure_reason=None
        )

    def add_step(
        self, 
        workflow_id: str,
        state: np.ndarray, 
        action: int,
        reward: float, 
        done: bool, 
        timestep: int,
        task_id: str,
        is_failure: bool,
        failure_reason: Optional[str]
    ):
        """内部：添加轨迹步骤"""
        if workflow_id not in self.buffers:
            self.buffers[workflow_id] = Trajectory(max_length=self.max_length)

        traj = self.buffers[workflow_id]
        traj.add_step(
            state=state, 
            action=action, 
            reward=reward, 
            done=done,
            timestep=timestep, 
            task_id=task_id,
            is_failure=is_failure, 
            failure_reason=failure_reason
        )
        # 工作流完成/失败时，结算轨迹
        if done:
            traj.compute_returns()
            self.completed_trajectories.append(traj.to_dict(workflow_id))
            del self.buffers[workflow_id]
        
    def finalize_all(self):
        """处理未完成/未失败的工作流"""
        for wf_id, traj in list(self.buffers.items()):
            if len(traj.states) == 0:
                continue
            if traj.dones and traj.dones[-1]:
                traj.dones[-1] = False
            traj.compute_returns()
            self.completed_trajectories.append(traj.to_dict(wf_id))
        self.buffers.clear()

    def save_all(self, path: str):
        """保存所有轨迹（成功+有效失败）"""
        self.finalize_all()
        
        # 统计成功/失败记录比例（便于调试）
        total_steps = sum(len(traj['states']) for traj in self.completed_trajectories)
        failure_steps = sum(np.sum(traj['is_failure_steps']) for traj in self.completed_trajectories)
        failure_ratio = failure_steps / total_steps if total_steps > 0 else 0.0
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.completed_trajectories, f)
        
        print(f"[TrajectoryCollector] 已保存 {len(self.completed_trajectories)} 条轨迹")
        print(f"[TrajectoryCollector] 总步骤数：{total_steps}，失败步骤数：{failure_steps}，失败比例：{failure_ratio:.2%}")
        # 采样阶段调试信息
        print(f"[TrajectoryCollector-DEBUG] intermediate_calls={self._debug_intermediate_calls}, "
              f"completed_calls={self._debug_completed_calls}, failed_calls={self._debug_failed_calls}")



