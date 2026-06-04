---
title: 分布式 Sharding 模式
scope: 使用 device_mesh + sharding + remote 实现多 GPU 协作
hardware: 多 GPU 系统，需要 FlagCX 通信库
prerequisites: api/distributed.md
---

# 分布式 Sharding 模式

## 核心思想

TLE 的分布式编程模型基于三个概念：

1. **Device Mesh** — 定义设备的逻辑拓扑
2. **Sharding** — 描述 tensor 如何分布在 mesh 上
3. **Remote Access** — 跨设备数据访问

## Cluster 内通信模式

适用于同一 cluster 内的 CTA 间通信（如 Hopper cluster launch）。

```python
@triton.jit
def cluster_kernel(input_ptr, output_ptr, ...):
    # 定义 cluster 拓扑
    mesh = tle.device_mesh({
        "block_cluster": [("cluster_x", 2), ("cluster_y", 2)]
    })

    # 获取当前 CTA 在 cluster 中的位置
    my_x = tle.shard_id(mesh, "cluster_x")
    my_y = tle.shard_id(mesh, "cluster_y")

    # 本地计算
    local_data = tl.load(input_ptr + ...)

    # 同步 cluster 内所有 CTA
    tle.distributed_barrier(mesh)

    # 访问邻居 CTA 的数据
    neighbor_x = (my_x + 1) % 2
    remote_data = tle.remote(local_data, neighbor_x, scope=mesh)

    # 合并结果
    result = local_data + remote_data
    tl.store(output_ptr + ..., result)
```

## Sharding 规格定义

```python
# 2D mesh: 数据并行 x 张量并行
mesh = tle.device_mesh({"dp": 4, "tp": 2})

# 矩阵 A: 沿行切分到 dp 轴
spec_a = tle.sharding(mesh, split=[tle.S("dp"), None])

# 矩阵 B: 沿列切分到 tp 轴
spec_b = tle.sharding(mesh, split=[None, tle.S("tp")])

# 输出: 需要沿 tp 轴 all-reduce
spec_out = tle.sharding(mesh, split=[tle.S("dp"), None], partial=[tle.P("tp")])
```

## Sharding 标记类型

| 标记 | 含义 | 用途 |
|------|------|------|
| `tle.S("axis")` | Split — 沿 mesh 轴切分 | 数据并行、张量并行 |
| `tle.P("axis")` | Partial — 需要 reduce | all-reduce 聚合 |
| `tle.B` | Broadcast — 所有设备复制 | 共享参数 |

## 通信基础设施

| 组件 | 说明 |
|------|------|
| `tle.distributed_barrier(mesh)` | 同步指定 mesh 范围内的设备 |
| `tle.remote(tensor, shard_id, scope)` | 读取远程设备数据 |
| FlagCX | 底层通信库（`third_party/tle/third_party/flagcx/`） |

## 当前限制

- `tle.distributed_dot` 尚未实现（deferred to M5）
- 分布式操作目前主要支持 cluster 级通信
- 跨节点通信依赖 FlagCX 的可用性

## 参考测试

- `python/test/tle/unit/test_tle_distributed.py`
- `python/test/tle/integration/test_tle_distributed.py`
