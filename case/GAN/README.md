# DCGAN 生成 MNIST 手写数字

用 PyTorch 实现的深度卷积生成对抗网络 (DCGAN),在 MNIST 数据集上训练,
生成逼真的手写数字图像。

## 文件说明

| 文件 | 说明 |
|---|---|
| `models.py` | 生成器 (Generator) 和判别器 (Discriminator) 的网络结构定义 |
| `train.py` | 训练脚本,自动下载 MNIST 数据并训练模型 |
| `generate.py` | 用训练好的生成器生成新图像 |
| `requirements.txt` | 依赖包列表 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python3 train.py --epochs 20 --batch_size 128
```

常用参数:
- `--epochs`: 训练轮数,默认 20(MNIST 上 20~30 轮通常就能看到不错的效果)
- `--batch_size`: 批大小,默认 128
- `--latent_dim`: 噪声向量维度,默认 100
- `--lr`: 学习率,默认 2e-4(DCGAN 论文推荐值,不建议改太大)
- `--out_dir`: 输出目录,默认 `./outputs`

训练时会自动:
- 把 MNIST 数据下载到 `./data` 目录(首次运行需要联网)
- 每个 epoch 在 `outputs/samples/` 下保存一张生成样本网格图,方便观察训练进展
- 训练结束后把模型权重保存到 `outputs/checkpoints/generator.pth` 和 `discriminator.pth`
- 保存 `outputs/loss_curve.png` 损失曲线图

### 2. 生成新图像

```bash
python3 generate.py --checkpoint ./outputs/checkpoints/generator.pth --num_images 64
```

会在 `outputs/generated.png` 生成一张包含 64 张随机手写数字的网格图。

## 网络结构

**生成器 (Generator)**:输入 100 维随机噪声 → 通过 3 层转置卷积逐步上采样 → 输出 28×28 灰度图像(Tanh 激活,像素范围 [-1, 1])

**判别器 (Discriminator)**:输入 28×28 图像 → 通过 3 层卷积逐步下采样提取特征 → 输出该图像为真实图像的 logit

## 训练原理简述

GAN 由两个网络博弈训练:
- **判别器 D** 学习区分真实图像和生成器伪造的图像(让 `D(real)` 接近 1,`D(fake)` 接近 0)
- **生成器 G** 学习生成能骗过判别器的图像(让 `D(G(z))` 接近 1)

两者交替训练,直到生成器能产出以假乱真的图像。

## 小提示

- 如果训练不稳定(loss 震荡剧烈、生成图像全是噪声或模式坍塌),可以尝试:
  - 降低学习率
  - 调整训练时 D 和 G 的更新频率比例
  - 检查 BatchNorm 是否生效、初始化是否正确(本代码已按 DCGAN 论文做了权重初始化)
- 想换成生成自己的图像数据集,只需把 `train.py` 里的 `datasets.MNIST(...)` 换成 `torchvision.datasets.ImageFolder(...)` 并调整图像尺寸/通道数(比如彩色图像通道数要从 1 改成 3)。
