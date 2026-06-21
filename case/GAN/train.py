"""
DCGAN 训练脚本:在 MNIST 数据集上训练生成器和判别器

运行方式:
    python3 train.py --epochs 20 --batch_size 128

训练过程中会:
    1. 每个 epoch 打印一次判别器/生成器的损失
    2. 定期保存生成的样本图像到 samples/ 目录
    3. 训练结束后保存模型权重到 checkpoints/ 目录
    4. 保存损失曲线图
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图片
import matplotlib.pyplot as plt

from models import Generator, Discriminator, weights_init


def parse_args():
    parser = argparse.ArgumentParser(description="训练 DCGAN 生成 MNIST 手写数字")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--latent_dim", type=int, default=100, help="噪声向量维度")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率 (DCGAN 论文推荐值)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam 优化器的 beta1 (DCGAN 论文推荐值)")
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST 数据存放目录")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="输出目录(样本图像、模型权重等)")
    parser.add_argument("--sample_interval", type=int, default=1, help="每隔多少个 epoch 保存一次生成样本")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 设备选择:有 GPU 用 GPU,没有就用 CPU ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- 输出目录准备 ----
    samples_dir = os.path.join(args.out_dir, "samples")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- 数据准备 ----
    # 把像素值从 [0, 1] 归一化到 [-1, 1],与生成器的 Tanh 输出对应
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    print(f"训练集大小: {len(train_dataset)}")

    # ---- 模型初始化 ----
    netG = Generator(latent_dim=args.latent_dim).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # ---- 损失函数与优化器 ----
    # BCEWithLogitsLoss = Sigmoid + BCELoss 合并,数值更稳定(判别器最后一层不再需要 Sigmoid)
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    # 固定一组噪声,用于在训练过程中持续观察同一批噪声生成的图像变化
    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    G_losses, D_losses = [], []

    print("开始训练...")
    for epoch in range(args.epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            bs = real_imgs.size(0)

            # =========================================
            # 第一步:训练判别器 D
            # 目标: 最大化 log(D(real)) + log(1 - D(G(z)))
            # =========================================
            netD.zero_grad()

            # -- 用真实图像训练 --
            label = torch.full((bs,), real_label, dtype=torch.float, device=device)
            output_real = netD(real_imgs)
            loss_D_real = criterion(output_real, label)
            loss_D_real.backward()

            # -- 用生成器伪造的图像训练 --
            noise = torch.randn(bs, args.latent_dim, device=device)
            fake_imgs = netG(noise)
            label.fill_(fake_label)
            # detach():判别器训练阶段不需要把梯度传回生成器
            output_fake = netD(fake_imgs.detach())
            loss_D_fake = criterion(output_fake, label)
            loss_D_fake.backward()

            loss_D = loss_D_real + loss_D_fake
            optimizerD.step()

            # =========================================
            # 第二步:训练生成器 G
            # 目标: 最大化 log(D(G(z))),即让判别器误判生成图像为真
            # =========================================
            netG.zero_grad()
            label.fill_(real_label)  # 生成器希望判别器把假图判为真(label=1)
            output = netD(fake_imgs)
            loss_G = criterion(output, label)
            loss_G.backward()
            optimizerG.step()

            # ---- 记录损失 ----
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                    f"D_loss: {loss_D.item():.4f}  G_loss: {loss_G.item():.4f}"
                )

            G_losses.append(loss_G.item())
            D_losses.append(loss_D.item())

        # ---- 每个 epoch 结束后保存生成样本 ----
        if (epoch + 1) % args.sample_interval == 0:
            netG.eval()
            with torch.no_grad():
                fake_samples = netG(fixed_noise).detach().cpu()
            vutils.save_image(
                fake_samples,
                os.path.join(samples_dir, f"epoch_{epoch+1:03d}.png"),
                normalize=True,
                nrow=8,
            )
            netG.train()
            print(f"  -> 已保存第 {epoch+1} 轮生成样本到 {samples_dir}")

    # ---- 保存模型权重 ----
    torch.save(netG.state_dict(), os.path.join(ckpt_dir, "generator.pth"))
    torch.save(netD.state_dict(), os.path.join(ckpt_dir, "discriminator.pth"))
    print(f"模型权重已保存到 {ckpt_dir}")

    # ---- 保存损失曲线 ----
    plt.figure(figsize=(8, 5))
    plt.title("生成器与判别器训练损失")
    plt.plot(G_losses, label="Generator", alpha=0.8)
    plt.plot(D_losses, label="Discriminator", alpha=0.8)
    plt.xlabel("迭代次数 (iterations)")
    plt.ylabel("损失 (Loss)")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=120, bbox_inches="tight")
    print(f"损失曲线已保存到 {os.path.join(args.out_dir, 'loss_curve.png')}")

    print("训练完成!")


if __name__ == "__main__":
    main()
