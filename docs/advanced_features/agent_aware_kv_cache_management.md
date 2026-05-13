# Agent 感知 KV Cache 管理优化设计

本文档汇总 SGLang 在 Agent 工作负载下的两类 KV Cache 管理优化，用于形成统一方案视图：

- **Continuum Pin KV（当前分支已实现）**：当一次请求触发 tool call 时，短时间保护对应前缀分支，提升后续连续请求的复用概率。
- **KV Demote for Context Compression（方案设计中）**：当 OpenClaw 触发上下文压缩时，对复用价值较低的旧分支和 summarization 临时分支做一次性降级，使其在缓存压力下优先被淘汰。

两类优化都不改变正常推理语义，只改变 SGLang 对不同 KV 分支的保留优先级。

## 一、方案说明

### 1.1 背景

在 Agent 场景中，前缀缓存价值并不总是等价，至少存在两类典型需求：

1. **短期高价值前缀保护**：tool call 之后，后续请求通常会沿当前会话的最近前缀继续生成，此时对应 KV 在短时间内具有较高复用价值。
2. **低价值前缀主动让渡**：上下文压缩后，被摘要替换的旧上下文分支，以及压缩请求自身生成的 summarization KV，后续复用概率较低，应尽快为新上下文让出缓存空间。

当前仓库中的 Continuum 调度已经覆盖第一类问题；本设计文档补充第二类问题，并将两者合并为统一的 Agent 感知 KV Cache 管理方案。

#### 1.1.1 Continuum Pin KV（已实现）

当模型输出中检测到 tool call 时，SGLang 可在短时间内保护当前请求命中的 RadixTree 节点，避免该前缀在工具调用往返期间被常规驱逐策略淘汰。

当前实现要点：

- 通过 `--schedule-policy continuum` 启用 Continuum 调度。
- OpenAI chat serving 层在检测到 tool call 后，向 scheduler 发送 `ContinuumPinReqInput`。
- scheduler 根据请求 `rid` 找到对应的 `last_node`，并对该节点执行一次额外的 `inc_lock_ref`。
- `ContinuumPinManager` 负责维护 pin 生命周期，到期后执行对称的 `dec_lock_ref`。
- 在 waiting queue 排序时，已 pin 的请求优先于普通请求，再按 FCFS 处理。

该机制本质上是对**高复用概率分支的临时保护**。

#### 1.1.2 上下文压缩 KV 降级（待补充实现）

OpenClaw 上下文窗口有限，当上下文长度超过阈值水位时，会触发上下文压缩。压缩对象不是完整上下文，而是历史上下文中的一段旧内容；该旧内容会被摘要替换，近期上下文通常保持不变。

在该场景下，有两类 KV Cache 复用价值较低：

1. **旧上下文中被压缩替换的 KV**：压缩后上下文结构已变化，该旧分支及其子分支无法被新上下文继续复用。
2. **压缩请求自身输入产生的 KV**：该请求是内部 summarization 请求，通常只服务本次压缩，不具备长期复用价值。

问题在于：这些 KV 会因压缩请求刷新访问时间，继续被 LRU 等驱逐策略保留，进而挤占新上下文可用的缓存空间。

目标：在不改变正常推理语义的前提下，由 OpenClaw 在压缩请求中向 SGLang 传递一次性降级 hint，SGLang 在本次请求处理中将低价值 KV 降级，使其在后续缓存压力下优先被驱逐。

### 1.2 环境配置

- 万全发版版本：龙虾湖
- 推理引擎：SGLang
- 目标模型：MiniMax M2.7（Target Model）
- 硬件要求：CUDA GPU + CPU

### 1.3 功能描述

#### 1.3.1 Continuum Pin KV

- 检测到 tool call 时，临时 pin 当前请求最后命中的 RadixTree 节点。
- pin 生命周期由 `continuum_pin_seconds` 控制，默认值见当前 `ServerArgs`。
- pin 期间，对应节点不会因常规缓存压力被优先淘汰。
- pin 到期后自动释放，不影响后续正常驱逐。

#### 1.3.2 被压缩上下文 KV 降级

OpenClaw 触发上下文压缩时，向 SGLang 传递压缩前后上下文信息，SGLang 基于 RadixTree 定位旧上下文中被压缩替换的 KV 分支，并按当前驱逐策略将其降级，使其在缓存压力下优先淘汰。

#### 1.3.3 压缩请求 KV 降级

SGLang 识别本次上下文压缩请求自身产生的临时 KV Cache，并在写入 RadixTree 后立即降级，避免内部 summarization 请求的 KV 长期占用缓存空间。

#### 1.3.4 分叉节点精细化处理

当旧上下文与压缩后上下文的分叉点落在 RadixTree 节点内部时，SGLang 根据失效 token 数量决定保留或拆分节点，在保证缓存命中率的同时减少无效 KV 占用。

## 二、方案设计

### 2.1 总体方案

统一思路是：**对高价值 KV 做短期保护，对低价值 KV 做一次性降级**。

#### 2.1.1 Continuum Pin 流程（已实现）

1. serving 层解析模型输出并检测 tool call。
2. 若调度策略为 `continuum`，则发送 `ContinuumPinReqInput(rid, seconds, min_protected_len)` 到 scheduler。
3. scheduler 根据 `rid` 找到当前请求最近一次命中的 cache node。
4. 若节点当前未被 pin，则执行一次 `inc_lock_ref`；若旧 pin 已过期，则先对称释放。
5. `ContinuumPinManager` 记录过期时间与保护信息。
6. 后续调度时，命中已 pin 节点的 waiting request 被优先调度。
7. pin 到期后，scheduler 周期性回收并执行 `dec_lock_ref`。

该流程的目标是让工具调用往返期间的热点前缀更稳定地留在缓存中。

#### 2.1.2 Context Compression KV Demote 流程（设计）

OpenClaw 在压缩请求中携带一次性 KV 降级 hint。SGLang 在处理该请求时：

1. 解析当前压缩请求 `messages`，生成 `current_key`，表示当前请求输入，即需要压缩的部分。
2. 解析 `system_messages`、`demote_messages`，生成 `system_key`、`demote_key`，表示完整的旧上下文。
3. 在 RadixTree 前缀匹配过程中，根据 `system_key`、`demote_key` 定位旧上下文失效分支。
4. 降级旧上下文被压缩分支及其所有子分支。
5. 压缩请求执行完成并写入 KV 后，降级当前压缩请求新增分支。

### 2.2 与现有缓存机制的关系

- **Pin** 通过提升保护等级来延缓驱逐，适用于短期热点前缀。
- **Demote** 不删除 KV，只降低其后续保留优先级，适用于结构上已经失效或业务价值低的前缀。
- 两种机制都依托现有 RadixTree / Prefix Cache 元数据，不改变请求语义与 KV 命中逻辑。
- 当两种策略同时存在时，应以请求生命周期内的显式保护约束为先，再由驱逐策略决定剩余节点的淘汰顺序。

### 2.3 接口设计

#### 2.3.1 已有 Continuum Pin 接口

内部请求结构：

- `ContinuumPinReqInput.rid`：需要保护的请求 ID。
- `ContinuumPinReqInput.seconds`：pin 持续时间。
- `ContinuumPinReqInput.min_protected_len`：最小保护前缀长度，当前分支中默认传 `0`。

触发方式：

- 当前由 chat serving 层在检测到 tool call 时自动发送。
- 仅在 `--schedule-policy continuum` 下生效。

#### 2.3.2 OpenClaw KV Demote 接口（设计）

OpenClaw 在压缩请求中增加 `openclaw_kv_demote` 字段：

| 字段 | 说明 |
| --- | --- |
| `system_messages` | 系统提示词上下文，不需要被降级部分 |
| `demote_messages` | 被压缩部分，对应 KV 需要被降级 |
| `demote_old_context_subtree` | 是否降级旧上下文被压缩分支 |
| `demote_current_request` | 是否降级当前压缩请求输入产生的 KV |
| `invalid_token_delete_threshold` | 共享前缀落在 RadixTree 节点内部时的失效 token 阈值，默认 256 |

## 三、接口和参数设计

### 3.1 已落地参数（Continuum Pin）

```bash
--schedule-policy continuum
--continuum-pin-seconds 3.0
```

示例：

```bash
python -m sglang.launch_server \
    --model-path /models/MiniMax-M2.7 \
    --schedule-policy continuum \
    --continuum-pin-seconds 3 \
    --host 0.0.0.0 \
    --port 30000
```

### 3.2 规划中参数（KV Demote）

```bash
--enable_demote_kv   # 使能 KV cache 降级功能
```

示例：

```bash
python -m sglang.launch_server \
    --model-path /models/MiniMax-M2.7 \
    --reasoning-parser minimax-append-think \
    --tool-call-parser minimax-m2 \
    --tp-size 8 \
    --ep-size 8 \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 12 \
    --enable_demote_kv
```

## 四、落地边界与实现状态

### 4.1 当前分支已实现内容

- Continuum 调度策略中的 request-level cache pin
- tool call 触发的自动 pin 信号下发
- pin 生命周期管理与过期释放
- waiting queue 中对 pinned request 的优先调度

### 4.2 当前分支未实现内容

- `openclaw_kv_demote` 请求字段解析
- 被压缩旧上下文子树定位与降级
- 当前 summarization 请求新增分支降级
- `--enable_demote_kv` 启动参数及其配套执行路径

## 五、其他说明

- 本文档描述的是一个整体方案，其中 Continuum Pin 已在当前仓库改动中落地，KV Demote 仍处于接口与流程设计阶段。
- 对外行为上，两类能力都不改变模型生成结果，仅调整缓存资源在 Agent 场景下的分配策略。
