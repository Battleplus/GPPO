# Adaptive Gate 公式与实现审计

## 审计结论

原论文 Eq.(1)–(3) 没有充分规定 `f_{i,j,k}` 的输出非线性、插入位置和
gate 后归一化方式，因此不存在唯一可由论文文字推出的实现。当前项目必须同时
保留“论文兼容 Literal 解释”和“实现歧义诊断变体”，不能把某个后验表现更好的
变体改称论文原式。

冻结定义如下：

```text
Literal-Paper / Adaptive-current
task_message + sigmoid + bias=0 + gate后不二次归一化
```

这个定义已经在 `configs/GATE_DIAGNOSTIC_PROTOCOL.json` 预注册，后续无论结果
好坏均不得修改名称或含义。

## 1. f(i,j,k) 当前作用在哪里

当前 Literal 实现先在 self 与合法 task neighbors 的同一域计算：

```text
alpha = softmax(score)
coefficient(task) = alpha(task) * sigmoid(f)
coefficient(self) = alpha(self)
```

因此 `f` 作用于 softmax 后的 task-message coefficient，而不是 attention score，
也不是聚合后的整个 UAV 表示。代码中的 `task_message` 名称准确描述了这个行为。

诊断变体的含义：

- `Adaptive-score`：对 task score 加 `log(gate)`，随后在 self/task 联合域做一次 softmax；
- `Adaptive-softplus`：位置仍是 task message，但输出允许超过1；
- `Adaptive-warmup`：前50轮临时强制 gate=1，随后回到 Literal task-message gate；
- `NoGate`：task coefficient 直接等于 alpha；
- `SingleHead`：独立的普通单头图注意力结构对照。

## 2. gate 后是否重新归一化

Literal-Paper 不重新归一化。这意味着 task gate 会改变总消息质量，聚合系数和不再
保证为1。这不是普通 attention reweighting，而是 attention 与消息幅值的联合调制。

`Adaptive-score` 明确重新归一化：gate 在 softmax 前改变竞争，最终系数和仍为1。
这两个版本回答不同问题，不能只用同一个“adaptive attention”标签混在一起。

## 3. sigmoid 是否只能衰减

是。sigmoid 输出严格位于 `(0,1)`，Literal task-message 版本只能保留或衰减 task
message，不能放大。self coefficient 不受 gate 影响，因此 gate 还会相对提高 self
信息占比。

`Adaptive-softplus` 用正且无上界的输出检查“只衰减”是否是性能瓶颈。softplus
不是论文默认；它只是预注册敏感性变体。

## 4. bias=0 是否使初始消息约减半

严格来说不是恒等于0.5：gate 是两层 MLP，虽然最后一层 bias=0，但其他权重和
输入会产生非零 logit。不过在对称随机初始化下，logit 的中心通常接近0，sigmoid
中心约为0.5，因此存在一开始系统性压低 task message 的风险。

已有300轮冻结 checkpoint 的 gate mean 为 `0.3517`，每 UAV 被移除的 attention
mass mean 为 `0.5632`。这证明问题不只是“初始化瞬间减半”：训练后的 Literal gate
仍在大幅衰减 task message。

预注册的 `Adaptive-bias2` 检查更接近恒等映射的初始化。两轮代码 Smoke 中，bias2
checkpoint 的 gate mean 约为 `0.8876`，符合预期，但这不是性能收益证据。

## 5. expected-RReLU 的影响

Literal 默认使用 RReLU 负斜率区间 `[0.125, 1/3]` 的确定性期望，而不是每次前向
随机采样。它有两个相反影响：

- 好处：同一状态/动作在 PPO old/new policy ratio 计算中保持确定，减少由随机激活
  引入的 ratio 噪声；
- 风险：去掉了原始 RReLU 的随机正则化，可能降低探索或改变论文训练动态。

因此 expected-RReLU 是为了可重复 PPO 比率作出的工程解释，不应表述为已证明等价于
论文训练。正式筛选保持 expected 模式，stochastic 模式只能作为补充敏感性分析，
不能读取 test100 后替换正式定义。

## 6. 额外参数与训练方差

gate MLP 为 `Linear(128,64) + ELU + Linear(64,1)`，新增参数：

```text
128×64 + 64 + 64×1 + 1 = 8,321
```

在只有单训练种子的情况下，额外参数可能带来：

- 更大的优化方差；
- validation20 选模方差；
- 更强的训练轮次敏感性；
- 通过消息幅值改变 critic/value scale；
- 更高的过拟合风险。

候选重评估已经观察到明显轮次敏感性：Adaptive 第100轮领先，但第150–300轮相对
NoGate/SingleHead 的方向反复变化。只有预注册三训练种子筛选才能区分结构收益与
偶然训练轨迹。

## 7. Warmup 的可审计语义

`Adaptive-warmup` 前50轮：

- task gate 精确强制为1；
- gate MLP 参数 `requires_grad=false`；
- 第51轮起恢复 learned sigmoid gate 与梯度。

正式300轮筛选中，第50轮强制 gate=1 的候选不具备最终 checkpoint 选模资格；该
修正记录在 `GATE_DIAGNOSTIC_PROTOCOL_AMENDMENT_001.json`，且声明未使用 test/test100。
20轮 Smoke 只检查机制链，不进行模型选择。

两轮实现 Smoke 已验证：

- 第1轮 gate 冻结，训练记录 gate gradient L2=`0`；
- 第2轮解冻，gate gradient L2=`0.001324`；
- policy entropy、approx KL、clip fraction 和 value loss 均正常落盘。

## 8. 当前允许的结论

可以确认：

- Literal Eq.(1)–(3) 的实现作用域已经明确并冻结；
- gate 不是常数且存在有效策略/PPO probe 梯度；
- 原 validation20 会影响 checkpoint 选择；
- 当前单种子结果不能支持 Adaptive 独立正收益。

尚不能确认：

- score、bias2、warmup 或 softplus 中哪一种稳定更好；
- expected-RReLU 与论文未知训练细节等价；
- Adaptive 额外参数在多种子下产生正收益而非更高方差。

这些问题只能由当前正在执行的预注册 Smoke 和后续三种子300轮筛选回答。
