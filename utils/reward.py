import sys,os
import math
import numpy as np
from typing import List, Dict, Any, Tuple
from utils.dependency import build_dag, is_dag_valid, get_topo_order

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from env.nodes import Master

def calculate_workflow_makespan(workflow_tasks: List[Dict], master) -> Dict:
    """
    计算单个工作流的makespan,定义为:
    所有任务的最大结束时间 - 所有任务的最小开始时间 + 任务间数据传输时间的最大值
    返回包含makespan和完成状态的字典
    """
    incomplete_tasks = [t for t in workflow_tasks if not t.get("is_completed", False)]
    if incomplete_tasks:
        return {
            "is_completed": False,
            "makespan": None,
            "error": f"存在未完成任务：{[t['task_id'] for t in incomplete_tasks]}",
            "incomplete_tasks": len(incomplete_tasks)
        }

    # 构建DAG并检查有效性
    dag = build_dag(workflow_tasks)
    if not is_dag_valid(dag):
        return {
            "is_completed": False,
            "makespan": None,
            "error": "循环依赖"
        }
    # 获取拓扑排序
    topo_order = get_topo_order(dag)
    if not topo_order:
        return {
            "is_completed": False,
            "makespan": None,
            "error": "无效的拓扑排序"
        }
    
    workflow_start_time = min(t["start_time"] for t in workflow_tasks)
    workflow_end_time = max(t["end_time"] for t in workflow_tasks)

    """
    计算DAG中总体数据传输时延:
    1. 确定每个任务在DAG中的层级
    2. 计算每一层内的最大传输时延（并行传输取最长）
    3. 逐层累加得到总体传输时延
    """
    # 映射任务ID到任务详情
    task_map = {t["task_id"]: t for t in workflow_tasks}

    # 1. 计算每个任务在DAG中的层级（从0开始）
    # 头节点层级为0，依赖任务的层级 = 所有依赖任务的最大层级 + 1
    task_level = {}
    for task_id in topo_order:
        deps = dag[task_id]
        if not deps:
            # 无依赖的头节点，层级为0
            task_level[task_id] = 0
        else:
            # 层级 = 所有依赖任务的最大层级 + 1
            max_dep_level = max(task_level[dep_id] for dep_id in deps)
            task_level[task_id] = max_dep_level + 1
    
    # 2. 按层级分组，收集每个层级的传输时延
    level_transfers = {}  # key: 层级, value: 该层级所有传输时延列表
    for task_id in topo_order:
        # task = task_map[task_id]
        deps = dag[task_id]
        current_level = task_level[task_id]
        
        if deps and current_level > 0:  # 非头节点才有传输时延
            # 初始化当前层级的传输时延列表
            if current_level not in level_transfers:
                level_transfers[current_level] = []
            
            # 计算当前任务与所有依赖之间的传输时延
            for dep_id in deps:
                transfer_time = master.calculate_transfer_time(
                    src_task_id=dep_id,
                    dst_task_id=task_id
                )
                level_transfers[current_level].append(transfer_time)

    # 3. 计算每一层的最大传输时延，并累加得到总体传输时延
    total_transfer_time = 0.0
    # 按层级升序处理（确保从第一层开始累加）
    for level in sorted(level_transfers.keys()):
        if level_transfers[level]:
            total_transfer_time += max(level_transfers[level])
            # 可选：打印每层的最大传输时延，便于调试
            # print(f"层级 {level} 最大传输时延: {max_level_transfer}")
    
    makespan = (workflow_end_time - workflow_start_time) + total_transfer_time

    return {
        "is_completed": True,
        "makespan": makespan,
        "error": None
    }

def calculate_reward(masters: List[Master]) -> List[float]:
    """
    核心目标：
    1. 最小化所有工作流的最大makespan（首要目标）
    2. 最大化工作流完成率（次要目标）
    """
    # 前置校验
    assert len(masters) == 3, f"需3个Master节点，实际传入{len(masters)}个"

    ########### 1. 收集所有工作流的makespan数据 ###########
    all_workflows = []  # 存储所有工作流的计算结果
    
    for master in masters:
        for workflow in master.workflows:
            tasks = workflow.get("tasks", [])
            if not tasks:
                continue  # 跳过空工作流
            
            # 计算单个工作流的makespan
            workflow_result = calculate_workflow_makespan(tasks, master)
            
            # 记录结果，包含所属Master信息
            all_workflows.append({
                "workflow_id": workflow.get("workflow_id", "unknown"),
                "master_id": master.id,
                **workflow_result
            })

    ########### 2. 计算核心指标 ###########
    # 2.1 工作流完成率
    total_workflows = len(all_workflows)
    completed_workflows = [w for w in all_workflows if w["is_completed"]]
    completion_rate = len(completed_workflows) / total_workflows if total_workflows > 0 else 0.0
    
    if not completed_workflows:
        # 无完成工作流时的惩罚
        return [0.0 for _ in masters]
    
    # 提取所有已完成工作流的makespan
    makespan_list = [w["makespan"] for w in completed_workflows]
    
    # 核心指标1：最大makespan（约束整体完成时间）
    max_makespan = max(makespan_list)
    
    # 核心指标2：平均makespan（反映整体效率）
    avg_makespan = np.mean(makespan_list)
    
    # 核心指标3：90分位makespan（过滤极端值，反映多数工作流表现）
    p90_makespan = np.percentile(makespan_list, 90)

    ########### 3. 多维度奖励计算 ###########
    # 缩放因子（根据实际场景调整）
    scale_max = 200.0    # 最大makespan的参考尺度
    scale_avg = 150.0    # 平均makespan的参考尺度
    scale_p90 = 180.0    # 90分位makespan的参考尺度
    
    # 3.1 最大makespan奖励（惩罚超长耗时，权重最高）
    max_reward = math.exp(-max_makespan / scale_max)  # 范围(0,1]
    
    # 3.2 平均makespan奖励（鼓励整体效率提升）
    avg_reward = math.exp(-avg_makespan / scale_avg)  # 范围(0,1]
    
    # 3.3 90分位makespan奖励（平衡极端值和整体表现）
    p90_reward = math.exp(-p90_makespan / scale_p90)  # 范围(0,1]
    
    # 3.4 完成率奖励
    completion_reward = completion_rate  # 范围[0,1]
    
    ########### 4. 奖励合并（权重按优先级分配） ###########
    # 权重设计逻辑：
    # - 最大makespan（40%）：确保不出现极端延迟
    # - 90分位makespan（30%）：关注多数工作流的表现
    # - 平均makespan（15%）：提升整体效率
    # - 完成率（15%）：保证工作流完成质量
    weights = {
        "max": 0.4,
        "p90": 0.3,
        "avg": 0.15,
        "completion": 0.15
    }
    
    base_reward = (
        max_reward * weights["max"] +
        p90_reward * weights["p90"] +
        avg_reward * weights["avg"] +
        completion_reward * weights["completion"]
    )

    ########### 5. 按Master分配奖励 ###########
    final_rewards = []
    for master in masters:
        master_workflows = [w for w in all_workflows if w["master_id"] == master.id]
        if not master_workflows:
            final_rewards.append(0.0)
            continue
        
        master_completed = [w for w in master_workflows if w["is_completed"]]
        if not master_completed:
            final_rewards.append(0.0)
            continue
        
        # 计算Master的局部多维度指标
        master_makespans = [w["makespan"] for w in master_completed]
        master_max = max(master_makespans)
        master_avg = np.mean(master_makespans)
        master_p90 = np.percentile(master_makespans, 90)
        
        # 局部优化因子（与全局指标对比）
        factor_max = 1 - (master_max / max_makespan)
        factor_p90 = 1 - (master_p90 / p90_makespan)
        factor_avg = 1 - (master_avg / avg_makespan)
        
        # 综合局部因子（限制波动范围±10%）
        local_factor = 1 + 0.1 * (factor_max + factor_p90 + factor_avg) / 3
        local_factor = max(0.9, min(local_factor, 1.1))  # 稳定奖励波动
        
        master_reward = base_reward * local_factor
        final_rewards.append(round(master_reward, 4))
    
    return final_rewards

def to_grid_rewards(node_reward: List[float]) -> np.ndarray:
    """将奖励转换为网格格式"""
    return np.array(node_reward).reshape([-1, 1])

def calculate_transfer_time(data_size_mb: float, src_node: dict, dst_node: dict) -> float:
    """
    从环境文件提取带宽，计算传输时间（秒）
    src_node/dst_node需包含："id"（节点ID，如edge-1-h1）、"type"（如edge_host/cloud_host）
    """
    # 1. 加载环境文件，提取固定带宽（四类链路带宽在文件中统一，直接定义）
    edge_intra_bw = 100    # 边缘集群内（文件中所有edge-x-hx间链路均为100Mbps）
    edge_inter_bw = 20     # 边缘集群间（文件中edge-x间链路均为20Mbps）
    cloud_edge_bw = 50     # 云边（文件中edge-x与cloud-1间链路均为50Mbps）
    cloud_intra_bw = 200   # 云内（文件中cloud-hx间带宽均为200Mbps）

    # 2. 从节点ID提取集群信息（如edge-1-h1→集群edge-1，cloud-h1→集群cloud-1）
    def get_cluster(node_id: str) -> str:
        if node_id.startswith("edge-"):
            return node_id.split("-h")[0]  # edge-1-h1 → edge-1
        elif node_id.startswith("cloud-"):
            return "cloud-1"  # 云节点统一归属cloud-1集群
        return ""

    src_cluster = get_cluster(src_node["id"])
    dst_cluster = get_cluster(dst_node["id"])
    src_type = src_node["type"]
    dst_type = dst_node["type"]

    # 3. 匹配链路类型，选择带宽
    if src_cluster and dst_cluster:
        # 边缘集群内（同edge集群，如edge-1-h1→edge-1-h2）
        if src_cluster.startswith("edge-") and src_cluster == dst_cluster:
            bandwidth = edge_intra_bw
        # 边缘集群间（不同edge集群，如edge-1-h1→edge-2-h2）
        elif src_cluster.startswith("edge-") and dst_cluster.startswith("edge-") and src_cluster != dst_cluster:
            bandwidth = edge_inter_bw
        # 云边（edge→cloud或cloud→edge，如edge-1-h1→cloud-h1）
        elif (src_cluster.startswith("edge-") and dst_cluster == "cloud-1") or (src_cluster == "cloud-1" and dst_cluster.startswith("edge-")):
            bandwidth = cloud_edge_bw
        # 云内（同cloud-1集群，如cloud-h1→cloud-h2）
        elif src_cluster == "cloud-1" and dst_cluster == "cloud-1":
            bandwidth = cloud_intra_bw
        else:
            bandwidth = 10  # 未知链路兜底
    else:
        bandwidth = 10  # 无效节点ID兜底

    # 4. 计算传输时间（MB→Mb×8，Mbps→秒÷带宽）
    if bandwidth <= 0 or data_size_mb < 0:
        return float("inf")
    return round(data_size_mb * 8 / bandwidth, 2)

if __name__ == "__main__":
    # 1. 定义节点（需包含id和type，与环境文件匹配）
    src = {"id": "edge-1-h1", "type": "edge_host"}  # 边缘1集群主机
    dst = {"id": "cloud-h2", "type": "cloud_host"}  # 云1集群主机

    # 2. 计算100MB数据的传输时间
    transfer_time = calculate_transfer_time(
        data_size_mb=100,
        src_node=src,
        dst_node=dst,
    )
    print(f"传输时间：{transfer_time} 秒")  # 输出：(100)/50=2.0 秒
