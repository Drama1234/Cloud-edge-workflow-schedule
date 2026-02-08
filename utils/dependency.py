import math
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict, deque

# def build_dag(tasks: List[Dict]) -> Dict[str, List[str]]:
#     """构建DAG依赖图（任务ID -> 直接依赖的任务ID列表）"""
#     task_ids = {t["task_id"] for t in tasks}
#     return {
#         t["task_id"]: [d for d in t["dependencies"] if d in task_ids]
#         for t in tasks
#     }

def build_dag(tasks):
    dag = {}
    # 自适应处理 Task / Dict 两种输入
    if isinstance(tasks[0], dict):
        for t in tasks:
            dag[t["task_id"]] = list(t["dependencies"])
    else:  # Task 对象
        for t in tasks:
            dag[t.task_id] = [d.task_id for d in t.dep_objs]
    return dag

def is_dag_valid(dag: Dict[str, List[str]]) -> bool:
    """检测DAG是否存在环"""
    visited = set()
    stack = set()

    def dfs(node):
        if node in stack:
            return False
        if node in visited:
            return True
        visited.add(node)
        stack.add(node)
        for d in dag.get(node, []):
            if not dfs(d):
                return False
        stack.remove(node)
        return True

    return all(dfs(n) for n in dag)


def get_topo_order(dag: Dict[str, List[str]]) -> List[str]:
    """正确统计入度+正确更新后驱入度"""
    in_degree = defaultdict(int)
    # 正确统计入度（节点的入度=其依赖列表的长度）
    for node, deps in dag.items():
        in_degree[node] = len(deps)  # 直接用依赖列表长度作为入度
        # 给依赖列表中不存在于DAG的节点初始化入度（避免KeyError）
        for dep in deps:
            if dep not in in_degree:
                in_degree[dep] = 0

    # 初始队列：入度为0的节点（真正可直接执行的节点）
    queue = deque([node for node in dag if in_degree[node] == 0])
    topo_order = []

    while queue:
        current = queue.popleft()
        topo_order.append(current)

        # 遍历所有节点，找“当前节点是其依赖”的后驱节点
        for node, deps in dag.items():
            if current in deps:  # 若当前节点是node的依赖
                in_degree[node] -= 1  # node的入度-1
                if in_degree[node] == 0:
                    queue.append(node)

    return topo_order

def calculate_longest_path(
    tasks: List[Dict[str, Any]],
    dag: Dict[str, List[str]],
    topo_order: List[str]
) -> Tuple[float, List[str]]:
    """
    计算DAG的最长路径（关键路径）
    参数:
        tasks: 任务列表
        dag: 依赖图（任务ID -> 依赖列表）
        topo_order: 拓扑排序结果
    返回:
        最长路径时间和路径上的任务ID列表
    """
    # 创建任务ID到任务的映射
    task_map = {task["task_id"]: task for task in tasks}
    
    # 存储每个节点的最长路径时间
    longest_time = {node: 0.0 for node in dag}
    # 存储前驱节点，用于重建路径
    predecessor = {node: None for node in dag}
    
    # 初始化：头节点的最长路径时间为自身持续时间
    for node in topo_order:
        if task_map[node]["is_head"]:
            longest_time[node] = task_map[node]["duration"]
    
    # 按拓扑顺序计算最长路径
    for node in topo_order:
        current_time = longest_time[node]
        # 找到所有依赖当前节点的后续节点（即当前节点是它们的依赖）
        successors = [n for n in dag if node in dag[n]]
        
        for succ in successors:
            # 后续节点的最长路径 = max(当前值, 当前节点路径 + 后续节点持续时间)
            if longest_time[succ] < current_time + task_map[succ]["duration"]:
                longest_time[succ] = current_time + task_map[succ]["duration"]
                predecessor[succ] = node
    
    # 找到最长路径的终点
    max_time = max(longest_time.values())
    end_node = [node for node, time in longest_time.items() if time == max_time][0]
    
    # 重建最长路径
    longest_path = []
    current_node = end_node
    while current_node is not None:
        longest_path.append(current_node)
        current_node = predecessor[current_node]
    
    # 反转路径以获得从起点到终点的顺序
    longest_path.reverse()
    
    return max_time, longest_path
