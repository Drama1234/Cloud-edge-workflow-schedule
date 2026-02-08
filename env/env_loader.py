import json
import os
from typing import Dict, Any, List

def load_env_from_json(file_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载云边环境配置（保持原始字段不变）
    
    原始字段说明:
    - 云主机：cpu_cores (CPU核心数)、ram_mb (内存MB)、bw_mbps (带宽Mbps)
    - 边缘集群：edge_clusters (而非edges)，包含latency_to_cloud_ms (到云延迟)
    - 链路：bw_mbps (带宽)、latency_ms (延迟)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"环境配置文件不存在: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            env_config = json.load(f)
        
        return env_config
    except json.JSONDecodeError as e:
        raise ValueError(f"解析环境配置JSON失败: {str(e)}") from e


def get_resource_capacities(env_config: Dict[str, Any]) -> Dict[str, Any]:
    """提取环境中各节点的资源容量信息（适配生成代码的字段）"""
    # 云中心资源（CPU核心数直接累加，无需转换MIPS）
    cloud_hosts = env_config["cloud"]["hosts"]
    capacities = {
        "cloud": {
            "total_cpu": sum(host["cpu_cores"] for host in cloud_hosts),
            "total_ram_mb": sum(host["ram_mb"] for host in cloud_hosts),
            "host_count": len(cloud_hosts)
        },
        "edge_clusters": []  # 对应生成代码的edge_clusters
    }
    
    # 边缘集群资源
    for edge in env_config["edge_clusters"]:
        edge_hosts = edge["hosts"]
        edge_cap = {
            "id": edge["id"],
            "total_cpu": sum(host["cpu_cores"] for host in edge_hosts),
            "total_ram_mb": sum(host["ram_mb"] for host in edge_hosts),
            "host_count": len(edge_hosts),
            "latency_to_cloud_ms": edge["latency_to_cloud_ms"]
        }
        capacities["edge_clusters"].append(edge_cap)
    
    return capacities


def print_env_summary(env_config: Dict[str, Any]):
    """打印环境配置摘要信息（适配生成代码的输出）"""
    capacities = get_resource_capacities(env_config)
    
    print("环境配置摘要:")
    print(f"云中心: {capacities['cloud']['host_count']} 个主机, 总CPU核心: {capacities['cloud']['total_cpu']}, 总内存: {capacities['cloud']['total_ram_mb']} MB")
    
    for i, edge in enumerate(capacities["edge_clusters"]):
        print(f"边缘集群 {edge['id']}: {edge['host_count']} 个主机, 总CPU核心: {edge['total_cpu']}, 总内存: {edge['total_ram_mb']} MB, 到云延迟: {edge['latency_to_cloud_ms']} ms")
    