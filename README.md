# 全掩码双 Decoder 方案  

> • Decoder-1：**Mask Picker**  
>   - 输入完整序列 x  
>   - 输出二值掩码 m¹，删去冗余 token → skeleton s = x ⊘ m¹  
>   - 目标：保留最能解释全句含义的“因”  
>   
> • Decoder-2：**Mask Inserter**  
>   - 输入 skeleton s  
>   - 在恰当位置插入 k 个 `[MASK]` → sequence Ŝ  
>   - 目标：给并行补全器充分的“问号”，由“因”推到“果”  
>   
> • Encoder + MLM head：一次性填充 Ŝ 中所有 `[MASK]` → x̂  
>   - 目标：x̂ ≈ 原句 x  

下面给出一套端到端可训练的完整流程，使  
1. Decoder-1 真正学会“删繁就简、因果归因”；  
2. Decoder-2 学会“根据因补果”；  
3. Encoder-MLM 负责细粒度 token 预测。  

---

## 1. 训练管线（单轮）

```
x            // 原始句
│
├─► (A) Decoder-1 : Mask-Picker
│      得到掩码 m¹   (可微采样)
│      skeleton s = Drop(x, m¹)     // 仅保留未被掩码的 token
│
├─► (B) Decoder-2 : Mask-Inserter
│      输出指针序列 p (长度 = k = |m¹| )      // 待插入位置
│      Ŝ = InsertMask(s, p)
│
├─► (C) Encoder + MLM
│      Ŝ ➜ x̂
│
└─► (D) 损失
       L_recon = CE(x̂ , x)                     // 重建
       L_comp  = α · |1 - retention|            // 压缩/保留率
       L_ptr   = β · |k - |m¹||                 // 覆盖一致
       总损失  L = L_recon + L_comp + L_ptr
```

### 1.1 可微离散操作  
* **Mask 选择 (Decoder-1)**  
  - 输出 per-token 概率 pᵢ，使用 Gumbel-Sigmoid 或 Straight-Through Bernoulli 采样 mᵢ∈{0,1}。  
* **Mask 插入 (Decoder-2)**  
  - 看作 Pointer Network：对 **s** 的片段间隙输出分布 qⱼ，  
    用 Gumbel-Softmax 采样插入索引，或者用 policy gradient。  
  - 插入数 k 固定为 |m¹|，用 L_ptr 约束。  

### 1.2 压缩正则 (信息瓶颈)  
```
L_comp = λ · max(0, r_keep - r_target)      // r_keep = mean(1-m¹)
```
促使 Decoder-1 尽量少保留；`r_target` 可从 5-20 % 逐渐退火。

---

## 2. 分阶段训练策略  

1. **Stage-0：预训练 Encoder-MLM**  
   * 常规随机 span corruption (T5/BERT) → 得到稳健的填充器。  
2. **Stage-1：只训 Decoder-2**  
   * 用人工 skeleton：随机删除 r_token% token；  
   * 训练它插回 `[MASK]`，让 Encoder-MLM 完全恢复。  
   * 此时 Decoder-2 拥有“因→果”能力。  
3. **Stage-2：端到端训 Decoder-1 + Decoder-2（MLM 冻结或低 lr）**  
   * 损失=上节 L；  
   * Decoder-1 学会掩去冗余，同时逼迫 Decoder-2 + MLM 仍然能重建。  
4. **Stage-3：微调三模块**  
   * 小学习率联合优化，提高极限质量。  

逐阶段可显著降低梯度方差，训练稳定。

---

## 3. 推理流程  

### 3.1 文本编辑  
```
prompt ─► Decoder-1 (可调 r_target)  
      skeleton ─► Decoder-2 ─► Ŝ ─► Encoder-MLM → edited_text
```
- 若用户希望“保留主体、润色细节”，调低 r_target (少删)；  
- 若希望“提炼摘要”，直接输出 skeleton。  

### 3.2 非自回归生成  
```
seed (<BOS>)  
repeat T times:            // T≈4-6
    seed ─► Decoder-2 ─► Ŝ ─► Encoder-MLM → new_seq
    seed ← new_seq
```
可选每一轮后执行 Decoder-1“再压缩”一次形成循环推理。  
整体路径=O(T) Encoder 前向，比自回归 O(n) latency 低。

---

## 4. 优缺点总结  

优点  
1. 纯 **掩码级** 操作，所有 token 生成都并行完成。  
2. Decoder-1/2 角色清晰：  
   * D1 = 提炼因（弃冗余）  
   * D2 = 根据因补果（提问位置）  
3. 可控性强：一句话通过 r_target 调摘要 vs 复原。  
4. 与 MaskGIT、UL2 等现有并行生成技术兼容（Encoder-MLM 直接复用）。  

风险 / 挑战  
1. 双重离散决策（选 mask + 插 mask）→ 训练方差大；直通估计器或 REINFORCE 需仔细调超参。  
2. 需额外 Pointer 级别位置编码，处理插空后位置偏移。  
3. 当 skeleton 过于稀疏时，Decoder-2 可能找不到可插点，导致重建失败；需 L_ptr 做强约束。  
4. 参数规模≈ (Decoder-1 + Decoder-2 + Encoder)；部署成本高于单 GPT 或单 BERT。  

---

## 5. 可行性结论  

• 只要选用可微采样技巧并采用“预训练-冻结-微调”分阶段策略，  
  双 Decoder 的**掩删 / 掩插**互补机制是可实现的。  
• Decoder-1 通过压缩正则真正学到“删繁就简”的归因能力；  
  Decoder-2 继承了自回归 Pointer 视角的“由因生果”能力；  
  Encoder-MLM 并行填词，保证整体生成质量与推理速度。  
• 关键落点在于离散位置决策的稳定训练与插删数量一致性控制——  
  解决这两点即可把方案做成一条具有解释性且高吞吐的 LLM 生产线。