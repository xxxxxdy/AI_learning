"""
DCGAN 模型定义:生成器 (Generator) 与判别器 (Discriminator)
用于在 MNIST 数据集上生成手写数字图像 (28x28 灰度图)
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    生成器:输入一个随机噪声向量 z (latent vector),输出一张 28x28 的伪造图像

    结构:用转置卷积 (ConvTranspose2d) 逐步上采样
    输入: (batch, latent_dim, 1, 1)
    输出: (batch, 1, 28, 28),像素值范围 [-1, 1] (配合 Tanh 激活)
    """

    def __init__(self, latent_dim: int = 100, feature_maps: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # 输入: (latent_dim, 1, 1) -> (feature_maps*4, 7, 7)
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # (feature_maps*4, 7, 7) -> (feature_maps*2, 14, 14)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # (feature_maps*2, 14, 14) -> (1, 28, 28)
            nn.ConvTranspose2d(feature_maps * 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # 输出归一化到 [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z 形状: (batch, latent_dim) -> reshape 为 (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """
    判别器:输入一张图像 (真实或生成的),输出该图像为"真实图像"的概率 (logit)

    结构:用卷积 (Conv2d) 逐步下采样,提取特征后判别真假
    输入: (batch, 1, 28, 28)
    输出: (batch, 1) 的 logit (未经过 Sigmoid,配合 BCEWithLogitsLoss 使用,数值更稳定)
    """

    def __init__(self, feature_maps: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            # (1, 28, 28) -> (feature_maps, 14, 14)
            nn.Conv2d(1, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps, 14, 14) -> (feature_maps*2, 7, 7)
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps*2, 7, 7) -> (1, 1, 1)
            nn.Conv2d(feature_maps * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.net(img)
        return out.view(-1, 1).squeeze(1)  # 展平为 (batch,)


def weights_init(m: nn.Module):
    """
    DCGAN 论文推荐的权重初始化方式:
    Conv 层权重用 N(0, 0.02) 初始化,BatchNorm 层权重用 N(1, 0.02) 初始化
    这能让训练更稳定
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # 简单自测:检查输入输出形状是否正确
    G = Generator(latent_dim=100)
    D = Discriminator()

    z = torch.randn(8, 100)
    fake_imgs = G(z)
    print("生成器输出形状:", fake_imgs.shape)  # 期望: (8, 1, 28, 28)

    out = D(fake_imgs)
    print("判别器输出形状:", out.shape)  # 期望: (8,)
