# -*- coding: utf-8 -*-
"""
简易 RNN（循环神经网络）模型 —— 纯 NumPy 实现
==============================================

包含：
1. RNN 前向传播
2. BPTT（Backpropagation Through Time）反向传播
3. 参数更新（梯度下降）
4. 一个字符级语言模型的训练示例（学习预测下一个字符）

结构说明：
    输入序列 x_1, x_2, ..., x_T
    隐藏状态: h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
    输出:     y_t = Why @ h_t + by
"""

import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        """
        初始化 RNN 参数

        参数:
            input_size:  输入向量维度（如词表大小，one-hot 编码）
            hidden_size: 隐藏层维度
            output_size: 输出向量维度（如词表大小）
        """
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 权重初始化（小随机数，避免梯度爆炸）
        self.Wxh = rng.standard_normal((hidden_size, input_size)) * 0.01   # 输入 -> 隐藏
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) * 0.01  # 隐藏 -> 隐藏
        self.Why = rng.standard_normal((output_size, hidden_size)) * 0.01  # 隐藏 -> 输出
        self.bh = np.zeros((hidden_size, 1))   # 隐藏层偏置
        self.by = np.zeros((output_size, 1))   # 输出层偏置

        # Adagrad 优化器所需的梯度平方累积缓存
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    def forward(self, inputs, h_prev):
        """
        前向传播

        参数:
            inputs: list[np.ndarray]，每个元素形状为 (input_size, 1) 的 one-hot 向量，长度为 T
            h_prev: 初始隐藏状态，形状 (hidden_size, 1)

        返回:
            xs, hs, ys, ps: 各时间步的输入、隐藏状态、输出（logits）、输出概率(softmax)
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)

        for t in range(len(inputs)):
            xs[t] = inputs[t]
            # 隐藏状态更新
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)
            # 输出层（未归一化的 logits）
            ys[t] = self.Why @ hs[t] + self.by
            # softmax 得到概率分布
            exp_y = np.exp(ys[t] - np.max(ys[t]))  # 数值稳定
            ps[t] = exp_y / np.sum(exp_y)

        return xs, hs, ys, ps

    def backward(self, xs, hs, ps, targets):
        """
        BPTT 反向传播，计算各参数梯度

        参数:
            xs, hs, ps: forward() 的输出
            targets: list[int]，每个时间步目标字符的索引

        返回:
            梯度字典 + 最后时刻隐藏状态（用于下一个 batch 续传）+ 总损失
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))

        loss = 0
        T = len(xs)

        for t in reversed(range(T)):
            # 交叉熵损失: -log(p_target)
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

            # 输出层梯度: dL/dy = p - one_hot(target)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            dWhy += dy @ hs[t].T
            dby += dy

            # 反传到隐藏层
            dh = self.Why.T @ dy + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh  # tanh 的导数

            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t - 1].T

            dh_next = self.Whh.T @ dh_raw

        # 梯度裁剪，防止梯度爆炸
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)

        grads = dict(dWxh=dWxh, dWhh=dWhh, dWhy=dWhy, dbh=dbh, dby=dby)
        return grads, hs[T - 1], loss

    def update_params(self, grads, lr=0.1):
        """使用 Adagrad 更新参数（比普通SGD更稳定，是RNN训练的常见选择）"""
        params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        mems = [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
        keys = ["dWxh", "dWhh", "dWhy", "dbh", "dby"]

        for param, mem, key in zip(params, mems, keys):
            dparam = grads[key]
            mem += dparam * dparam
            param -= lr * dparam / np.sqrt(mem + 1e-8)

    def sample(self, h, seed_idx, n, ix_to_char):
        """
        从训练好的模型中采样生成文本

        参数:
            h: 初始隐藏状态
            seed_idx: 起始字符的索引
            n: 生成字符数
            ix_to_char: 索引 -> 字符 的映射
        """
        x = np.zeros((self.input_size, 1))
        x[seed_idx] = 1
        indices = []

        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)

            idx = np.random.choice(range(self.input_size), p=p.ravel())
            x = np.zeros((self.input_size, 1))
            x[idx] = 1
            indices.append(idx)

        return "".join(ix_to_char[i] for i in indices)


def train_char_rnn(text, hidden_size=64, seq_length=25, lr=0.1, n_iters=2000):
    """
    用一段文本训练字符级 RNN，让它学习"预测下一个字符"

    参数:
        text: 训练用的文本语料
        hidden_size: 隐藏层维度
        seq_length: 每次 BPTT 展开的序列长度
        lr: 学习率
        n_iters: 训练迭代次数
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    print(f"语料长度: {len(text)} 字符, 词表大小: {vocab_size}")
    print(f"字符集: {chars}\n")

    rnn = SimpleRNN(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size)

    n, p = 0, 0
    h_prev = np.zeros((hidden_size, 1))
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # 初始损失基线

    while n < n_iters:
        # 如果数据读完了，或者是第一次迭代，重置隐藏状态
        if p + seq_length + 1 >= len(text) or n == 0:
            h_prev = np.zeros((hidden_size, 1))
            p = 0

        # 准备输入/目标序列（one-hot 编码）
        input_chars = text[p: p + seq_length]
        target_chars = text[p + 1: p + seq_length + 1]

        inputs = []
        for ch in input_chars:
            vec = np.zeros((vocab_size, 1))
            vec[char_to_ix[ch]] = 1
            inputs.append(vec)
        targets = [char_to_ix[ch] for ch in target_chars]

        # 前向 + 反向传播
        xs, hs, ys, ps = rnn.forward(inputs, h_prev)
        grads, h_prev, loss = rnn.backward(xs, hs, ps, targets)
        rnn.update_params(grads, lr=lr)

        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        if n % 200 == 0:
            sample_text = rnn.sample(h_prev, char_to_ix[input_chars[0]], 80, ix_to_char)
            print(f"[迭代 {n:5d}] loss={smooth_loss:.4f}")
            print(f"  采样输出: {sample_text!r}\n")

        p += seq_length
        n += 1

    return rnn, char_to_ix, ix_to_char


if __name__ == "__main__":
    # 一段简单的重复性文本，便于RNN快速学到规律
    text = "hello world! this is a simple rnn implemented from scratch using numpy. " * 50

    rnn, char_to_ix, ix_to_char = train_char_rnn(
        text, hidden_size=64, seq_length=25, lr=0.1, n_iters=2000
    )

    print("=" * 60)
    print("训练完成，最终生成示例：")
    h = np.zeros((rnn.hidden_size, 1))
    print(rnn.sample(h, char_to_ix["h"], 200, ix_to_char))
