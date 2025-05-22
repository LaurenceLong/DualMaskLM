以下用统一的符号把你补充的「多轮循环」版本写清楚，并给出可微实现、门控、以及 **Edit / Generation** 两条 Path 的自动切换办法。

---

## 1. 记号

* `X₀`        ：原始序列  
* `L`         ：最多循环轮数（训练时可设 3–6，推理时可早停）  
* `D₁`        ：Mask-Picker（删除）  
* `D₂`        ：Mask-Inserter（并行插位）  
* `E`         ：Encoder （预测`[MASK]`位置的logits）  
* `H`         ：MLM head（多轮填充）  
* `Gate`    ：标量门控 g∈[0,1]，控制本轮是否调用 D₁  
  g≈1 → **编辑**（先删后补），g≈0 → **生成**（直接补）

---

## 2. 训练流程（展开 L 轮，端到端反传）

```text
X_0 = 原句或随机起始序列
for l = 0 .. L-1:

    g_l = Gate(E_l) if l > 0 else 0  # 判定本轮是否“编辑”，第一轮强制走生成分支
    
    # ---------- 删除阶段 ----------
    if g_l > τ_edit:                 # τ_edit≈0.5，可软硬两用
        p = D₁(X_l)                  # token 保留概率
        m = Bernoulli_ST(p)          # Straight-Through 采样 0/1
        S_l = Drop(X_l, m)           # skeleton = 留下来的 token
    else:                            # 跳过编辑
        S_l = X_l

    # ---------- 插空阶段 ----------
    q = D₂(S_l)                      # 对 S_l 的 gap 概率
    idx = Pointer_ST(q, k=‖X_l‖-‖S_l‖)  # 可微指针采样
    Ŝ_l = InsertMask(S_l, idx)       # 带 [MASK] 的序列

    # ---------- 填充阶段 ----------
    E_l = E(Ŝ_l)                 # 预测logits
    X_{l+1} = H(E_l)             # 多轮填充
endfor
```

### 损失

```
L_recon  = CE(X_L , X_target)            // 最后一步与目标对齐
L_comp   = KL(p  ‖ Ber(r_target(g_l)))   // 控制删除量
L_ptr    = KL(q  ‖ Uniform_gap)          // 避免扎堆插位
L_gate   = c · g_l                       // “编辑”有代价
总损失   = L_recon + λ₁L_comp + λ₂L_ptr + λ₃L_gate
```

* `r_target(g)=g·r_max`，使 Gate 决定每轮保留率  
* `L_gate` 让模型在**编辑少但能重建**时倾向走生成分支

梯度可通过 Straight-Through trick 跨越多轮反向传播；内存不足时可用 **truncated BPTT** 或 REINFORCE。

---

## 3. 推理

### 3.1 Edit（用户已有文本）

```text
X_0 = 用户 prompt
for l=0..L-1:
    g_l = Gate(X_l)          # 若想纯编辑，可手动设 g_l=1
    ... 同上 ...
    if StopCriterion(X_{l+1}, X_l): break   # 置信度↑或变化↓
return X_{l+1}
```

### 3.2 Generation（从空或短 prompt 续写）

```text
X_0 = prompt  (可为空，仅 <BOS>)
for l=0..L-1:
    g_l = 0                  # 强制走生成分支
    S_l = X_l
    q = D₂(S_l) ; Ŝ_l=...
    X_{l+1} = E(Ŝ_l)
    if StopCriterion(...): break
return X_{l+1}
```

用户只需告诉模型“edit / generate / auto”。  
* `edit`  ⇒ g_l=1  
* `generate` ⇒ g_l=0  
* `auto` ⇒ 由 Gate 网络自己输出 g_l。

---

## 4. 关键实现要点

1. **Gate 训练**  
   无监督即可：`L_gate = c·g` 把“调用 D₁”视作要付费的动作。  
   * 若删除对 `L_recon` 贡献不大，模型就学会把 g 压低→生成。  
   * 若必须靠原文帮忙，g 会被拉高→编辑。  
   之后可用 RL 微调 Gate 以优化人工指标（ROUGE、BLEU、人工评分）。

2. **梯度稳定**  
   * Gumbel-Sigmoid / Gumbel-Softmax + Straight-Through  
   * 可先 **冻结 E**，只训 D₂，再端到端 joint。  
   * 对多轮循环，可先训练 1 轮，逐步放开到 L 轮（curriculum）。

3. **早停 StopCriterion**  
   * MaskGIT 式置信度平均值 > τ  
   * 或两轮输出差异 < ε（Levenshtein Transformer 的做法）  
   这样生成任务不会白跑 edit 分支，编辑任务不会过度迭代。

4. **位置漂移处理**  
   插空后 token index 变化：对 S_l 使用 **相对偏移位置编码**  
   `pos'(i) = pos_orig(i) + #MASK_inserted_left(i)`  
   保证 D₂ 在多轮中仍能准确定位 gap。

---

## 5. 小结

1. 用一个可学习 **Gate** 在“删除-插-补”完整链条和“直接插-补”之间切换，即把 **编辑** 与 **生成** 统一起来。  
2. 训练时把 L 轮全部展开即可端到端反传；StopCriterion 让推理时动态早停。  
3. 损失中增加「调用编辑的代价」即可让 Gate 自主权衡；用户仍可强制 override。  

这样 DualMaskLM 既能像 Grammarly 那样局部润色，又能像 GPT 那样自由写作，而且仍保持并行生成的低延迟优势。