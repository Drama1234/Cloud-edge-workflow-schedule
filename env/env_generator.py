import json
import os
import time

# h1:2cpu,4G
# h2:4cpu,16G
# h3:8cpu,32G

def generate_env_json(out_path: str):
    """
    新版环境 JSON 生成器：
    云端不再包含硬件字段（云端由 create_env_from_json 构造为 super-cloud）
    JSON 只负责网络结构（云-边缘延迟、边缘内部与跨边边延迟）
    JSON 只负责边缘主机硬件配置
    """
    # ========================
    # 1. 边缘集群
    # ========================
    edge_clusters = []
    for i in range(1, 4):
        latency = 60 if i == 1 else 80 if i == 2 else 100
        edge_clusters.append({
            "id": f"edge-{i}",
            "latency_to_cloud_ms": latency,
            "hosts": [
                {"id": f"edge-{i}-h1", "cpu_cores": 2, "ram_mb": 4096, "bw_mbps": 100},
                {"id": f"edge-{i}-h2", "cpu_cores": 2, "ram_mb": 4096, "bw_mbps": 100},
                {"id": f"edge-{i}-h3", "cpu_cores": 4, "ram_mb": 16384, "bw_mbps": 100},
                {"id": f"edge-{i}-h4", "cpu_cores": 4, "ram_mb": 16384, "bw_mbps": 100}
            ]
        })
    # ========================
    # 2. 链路结构
    # ========================
    links = []
    cloud_id = "cloud-super-0"
    edge_ids = [f"edge-{i}" for i in range(1, 4)]
    # 2.1 边缘集群内链路
    for edge in edge_clusters:
        hosts = edge["hosts"]
        for i in range(len(hosts)):
            for j in range(i + 1, len(hosts)):
                links.append({
                    "id": f"link-{hosts[i]['id']}-{hosts[j]['id']}",
                    "src": hosts[i]["id"],
                    "dst": hosts[j]["id"],
                    "bw_mbps": 100,
                    "latency_ms": 1
                })
    # 2.2 边缘集群间链路
    edge_latency = {(1, 2): 30, (1, 3): 60, (2, 3): 40}
    for i in range(3):
        for j in range(i + 1, 3):
            links.append({
                "id": f"link-{edge_ids[i]}-{edge_ids[j]}",
                "src": edge_ids[i],
                "dst": edge_ids[j],
                "bw_mbps": 20,
                "latency_ms": edge_latency[(i + 1, j + 1)]
            })
    # 2.3 云边链路
    for edge in edge_clusters:
        links.append({
            "id": f"link-{edge['id']}-{cloud_id}",
            "src": edge["id"],
            "dst": cloud_id,
            "bw_mbps": 50,
            "latency_ms": edge["latency_to_cloud_ms"]
        })
    # ========================
    # 3. 组装环境 JSON
    # ========================
    env_cfg = {
        "metadata": {
            "generated_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Cloud-Edge Environment for RL/DT (Super Cloud Version)"
        },
        "cloud_id": cloud_id,
        "edge_clusters": edge_clusters,
        "links": links
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_cfg, f, ensure_ascii=False, indent=2)

    print(f"[generate_env_json] Saved: {out_path}")
    print("Format: super-cloud + 3 edge clusters")

# 直接运行即可生成配置，无需输入参数
if __name__ == "__main__":
    generate_env_json("/root/autodl-fs/edgecloud/pretrain/datasource/env_cloud_edge1.json")
