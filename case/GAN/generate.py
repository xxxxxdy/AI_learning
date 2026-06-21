"""
用训练好的生成器生成新的手写数字图像

运行方式:
    python3 generate.py --checkpoint ./outputs/checkpoints/generator.pth --num_images 64
"""
import argparse
import os

import torch
import torchvision.utils as vutils

from models import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="用训练好的 DCGAN 生成器生成手写数字图像")
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/generator.pth", help="生成器权重路径")
    parser.add_argument("--latent_dim", type=int, default=100, help="噪声向量维度(需与训练时一致)")
    parser.add_argument("--num_images", type=int, default=64, help="生成图像数量")
    parser.add_argument("--output", type=str, default="./outputs/generated.png", help="输出图像保存路径")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator(latent_dim=args.latent_dim).to(device)
    netG.load_state_dict(torch.load(args.checkpoint, map_location=device))
    netG.eval()

    noise = torch.randn(args.num_images, args.latent_dim, device=device)
    with torch.no_grad():
        fake_imgs = netG(noise).cpu()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    vutils.save_image(fake_imgs, args.output, normalize=True, nrow=8)
    print(f"已生成 {args.num_images} 张图像,保存至: {args.output}")


if __name__ == "__main__":
    main()
