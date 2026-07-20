import argparse
import os
import warnings

import clip
import torch
from torch.utils.data import DataLoader

import data
from utils import DATASET_DOMAINS

warnings.filterwarnings("ignore", category=UserWarning, message="1Torch was not compiled with flash attention.")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CLIP image features")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Office", "OfficeHome", "VisDA", "DomainNet"])
    parser.add_argument("--domain", type=str, default=None, help="Single domain; default: all domains")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clip-path", type=str, default="ckpt/clip/ViT-B-16.pt")
    parser.add_argument("--output-root", type=str, default="./representations")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    domains = [args.domain] if args.domain else DATASET_DOMAINS[args.dataset]

    model, _ = clip.load(args.clip_path, device=device)
    model.eval()

    os.makedirs(os.path.join(args.output_root, args.dataset), exist_ok=True)

    with torch.no_grad():
        for domain in domains:
            save_dir = os.path.join(args.output_root, args.dataset, domain)
            os.makedirs(save_dir, exist_ok=True)

            dataset = data.UniDA_dataset(args.dataset, domain)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            count = 0
            for images, _ in loader:
                images = images.to(device)
                features = model.encode_image(images).cpu().float()
                for i in range(features.size(0)):
                    torch.save(features[i], os.path.join(save_dir, f"{count}.pt"))
                    count += 1

            print(f"Saved {count} features to {save_dir}")


if __name__ == "__main__":
    main()
