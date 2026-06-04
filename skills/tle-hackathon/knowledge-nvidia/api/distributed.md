---
title: TLE Distributed API
scope: 多 GPU 分布式计算 — mesh 定义、tensor sharding、远程访问
hardware: 多 GPU 系统，需要 FlagCX 通信库
prerequisites: api/core.md, 理解 multi-GPU 编程概念
---

# TLE Distributed API

入口：`triton.experimental.tle`（顶层导出）

## tle.device_mesh

定义设备拓扑，描述多 GPU 的逻辑排列。

```python
tle.device_mesh(topology=None) -> device_mesh
```

- `topology`: 可选的 `Mapping[str, Any]`，描述设备拓扑
- 返回 `device_mesh` 对象，用于后续 sharding 和通信操作

**典型用法：**
```python
mesh = tle.device_mesh({"x": 2, "y": 4})  # 2x4 mesh
```

## tle.sharding

定义 tensor 的分片策略。

```python
tle.sharding(mesh, split=None, partial=None) -> ShardingSpec
```

- `mesh`: `device_mesh` 对象
- `split`: 指定各维度如何切分到 mesh 轴上
- `partial`: 指定哪些维度需要 reduce

**辅助类型：**
- `tle.S` — Split 标记，表示沿某个 mesh 轴切分
- `tle.P` — Partial 标记，表示需要 all-reduce
- `tle.B` — Broadcast 标记，表示该维度在所有设备上复制

**典型用法：**
```python
spec = tle.sharding(mesh, split=[tle.S("x"), None], partial=[tle.P("y")])
```

## tle.ShardingSpec

Sharding 规格的数据类（frozen dataclass）。

```python
@dataclass(frozen=True)
class ShardingSpec:
    mesh: device_mesh
    split: ...
    partial: ...
    broadcast: ...
```

## tle.shard_id

获取当前设备在 mesh 某个轴上的 ID。

```python
tle.shard_id(mesh, axis) -> tl.tensor
```

- `mesh`: `device_mesh` 对象
- `axis`: 轴名称（str）或索引（int）
- 返回标量 int32 tensor（当前设备在该轴上的位置）

## tle.remote

访问远程设备上的 tensor 数据。

```python
tle.remote(tensor, shard_id, scope=None) -> tensor
```

- `tensor`: 本地 tensor
- `shard_id`: 目标设备的 shard ID
- `scope`: 可选的 `device_mesh`，限定通信范围
- 编译器生成 `tle.remote_pointers` IR op

## tle.distributed_barrier

分布式同步屏障。

```python
tle.distributed_barrier(mesh=None) -> None
```

- `mesh`: 可选的 `device_mesh`，指定同步范围
- 编译器生成 `tle.distributed_barrier` IR op
- 底层通过 FlagCX 通信库实现

## tle.distributed_dot (未完成)

分布式矩阵乘法。

```python
tle.distributed_dot(a: ShardedTensor, b: ShardedTensor, c=None)
# 当前状态：raise NotImplementedError("distributed_dot is deferred to M5")
```

## tle.make_sharded_tensor

将现有 handle 包装为带 sharding 信息的 `ShardedTensor`。

```python
tle.make_sharded_tensor(handle, sharding, shape=None) -> ShardedTensor
```

- `handle`: 底层 tensor/buffer
- `sharding`: 一个 `ShardingSpec` 实例，描述分片策略
- `shape`: 可选的正整数 tuple；若提供，长度必须与 `sharding.split` 的 rank 一致

## tle.ShardedTensor

带 sharding 标记的 tensor 包装类。

**属性：**

| 属性 | 类型 | 说明 |
|------|------|------|
| `handle` | tensor/buffer | 底层数据 |
| `sharding` | `ShardingSpec` | 分片规格 |
| `shape` | `tuple[int, ...] \| None` | 可选的全局 shape |

## tle.reshard

将 tensor 按新的 sharding spec 重新分片。

```python
tle.reshard(tensor, spec) -> ShardedTensor
# 当前状态：raise NotImplementedError("reshard is deferred to M4")
```

- `tensor`: 已有的 `ShardedTensor`
- `spec`: 目标 `ShardingSpec`

## 通信基础设施

| 组件 | 位置 |
|------|------|
| FlagCX 通信库 | `third_party/tle/third_party/flagcx/` |
| Remote pointers lowering | `tle.remote_pointers` IR op |
| Barrier lowering | `tle.distributed_barrier` IR op |

## 典型分布式 kernel 骨架

```python
@triton.jit
def distributed_kernel(input_ptr, output_ptr, ...):
    mesh = tle.device_mesh({"dp": num_devices})
    my_id = tle.shard_id(mesh, "dp")

    # 本地计算
    local_data = tl.load(input_ptr + ...)
    result = some_computation(local_data)

    # 访问远程数据
    neighbor_id = (my_id + 1) % num_devices
    remote_data = tle.remote(result, neighbor_id, scope=mesh)

    # 同步
    tle.distributed_barrier(mesh)

    tl.store(output_ptr + ..., result)
```
