# Gate 三种子 Smoke 报告

## 范围

- 场景：T5-10-48
- 变体：7个
- 训练种子：1、2、3
- 轮数：20
- rollout / batch / epochs：512 / 512 / 4
- checkpoint validation：validation100-A
- 后处理机制检查：validation-A 每checkpoint 5实例
- Gate诊断：每个Adaptive checkpoint 2实例 + 512步 PPO probe
- test/test100：未使用

启动时预注册协议 SHA256：
`0b863bf37045fa3287268c6f7695c439c529d4ca075032df3c04ce2d87f68f13`。

## 验收结论

`outputs/gate_screening/smoke/SMOKE_AUDIT.json`：`valid=true`。

全部通过：

- 21个预注册作业和checkpoint齐全；
- 所有训练历史到第20轮；
- validation split为validation-A且记录100实例；
- policy entropy、approx KL、clip fraction、actor/value loss、gate gradient均为有限值；
- 21个checkpoint后处理评估非法动作均为0，任务均完成；
- 非warmup Adaptive训练期gate gradient至少一次大于0；
- warmup三种子前20轮gate保持冻结，训练期gate gradient均为0；
- 15个Adaptive checkpoint重载后gate均非常数且PPO probe gradient大于0；
- 未生成或读取test结果。

因此允许进入三种子300轮筛选。Smoke只证明机制链可运行，不用于选择变体。

## validation-A 小样本后处理结果

下表为checkpoint重载后的5实例机制检查，只用于发现崩溃/非法动作，不用于排名：

| 变体 | seed1 | seed2 | seed3 | 三种子简单平均 |
|---|---:|---:|---:|---:|
| Adaptive-current | 18.2410 | 19.6138 | 18.0089 | 18.6213 |
| Adaptive-bias2 | 19.3339 | 17.0105 | 18.3981 | 18.2475 |
| Adaptive-warmup | 17.4783 | 19.4002 | 17.5620 | 18.1468 |
| Adaptive-score | 19.1882 | 17.9739 | 18.0792 | 18.4138 |
| Adaptive-softplus | 18.5751 | 19.4652 | 17.7575 | 18.5993 |
| NoGate | 17.8159 | 19.4002 | 17.5620 | 18.2594 |
| SingleHead | 18.0453 | 18.9451 | 17.5114 | 18.1673 |

5实例方差很大，禁止据此判断任一变体优胜。

## Gate分布诊断

20轮后不同种子的gate差异明显：

- Adaptive-current gate mean：0.8967 / 0.6412 / 0.8682；
- Adaptive-bias2：0.9617 / 0.9109 / 0.9681；
- Adaptive-warmup重载后：0.8795 / 0.8723 / 0.8847；
- Adaptive-score：0.7445 / 0.6817 / 0.8116；
- Adaptive-softplus：3.0268 / 1.5379 / 2.6854。

这说明：

- bias2确实使早期sigmoid gate更接近1；
- softplus确实产生大于1的消息放大，而不是只做衰减；
- current在20轮时尚未出现冻结300轮checkpoint中gate mean=0.3517的强衰减，gate行为会随训练阶段显著变化；
- 单个训练种子不足以描述gate分布。

## Warmup重载边界

Smoke的Adaptive-warmup在1–20轮训练和validation时强制gate=1，因此训练期行为等价于
NoGate；checkpoint只保存参数，不保存瞬时`force_gate_one`运行标志。后处理重载后会使用
bias2 learned gate，所以后处理5实例值不应与训练期validation混为一谈。

正式300轮筛选会在第51轮解冻，并依据编号amendment禁止第50轮warmup checkpoint参与
最终选模。该修正未读取test/test100，也不改变网络结构或超参数。

## 下一步

启动预注册三种子300轮筛选。只用validation-A选择候选；validation-B和test100必须等
全部训练和选择冻结后才执行。筛选报告必须保留所有负结果和种子级方向差异。
