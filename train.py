import argparse
import warnings
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import UniDA_lastlayerfeature, split_com_spvt_tpvt
from model import CASEClassifier, CASETrainer
from utils import DATASET_DOMAINS, DATASET_SPLITS, set_all_seeds, worker_init_fn

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CASE for Universal Domain Adaptation")
    parser.add_argument("--dataset", type=str, default="OfficeHome",
                        choices=["Office", "OfficeHome", "VisDA", "DomainNet"])
    parser.add_argument("--source", type=str, default=None, help="Source domain name")
    parser.add_argument("--target", type=str, default=None, help="Target domain name")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-k", type=int, default=2)
    parser.add_argument("--eval-last", type=int, default=10, help="Average H-score over last N epochs")
    return parser.parse_args()


def build_class_split(dataset_name):
    n_shared, n_source_private, n_total = DATASET_SPLITS[dataset_name]
    return split_com_spvt_tpvt(n_shared, n_source_private, n_total)


def compute_h_score(per_class_correct, per_class_num, num_common):
    per_class_acc = per_class_correct / (per_class_num + 1e-8)
    common_acc = per_class_acc[:num_common].mean()
    unknown_acc = per_class_acc[num_common]
    h_score = 2 * common_acc * unknown_acc / (common_acc + unknown_acc + 1e-8)
    return common_acc, unknown_acc, h_score


def run_task(args, source_domain, target_domain, device):
    classes_set = build_class_split(args.dataset)

    source_train_ds = UniDA_lastlayerfeature(
        args.dataset, source_domain, filter=lambda x: x in classes_set["source_classes"]
    )
    target_train_ds = UniDA_lastlayerfeature(
        args.dataset, target_domain, filter=lambda x: x in classes_set["target_classes"]
    )
    target_test_ds = UniDA_lastlayerfeature(
        args.dataset, target_domain, filter=lambda x: x in classes_set["target_classes"]
    )

    freq = Counter(source_train_ds.labels)
    source_weights = [1.0 / freq[x] for x in source_train_ds.labels]
    sampler = WeightedRandomSampler(source_weights, num_samples=len(source_train_ds.labels), replacement=True)

    source_train_dl = DataLoader(
        source_train_ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
        sampler=sampler, num_workers=0, worker_init_fn=worker_init_fn,
    )
    target_train_dl = DataLoader(
        target_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=0, worker_init_fn=worker_init_fn,
    )
    target_test_dl = DataLoader(
        target_test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, worker_init_fn=worker_init_fn,
    )

    num_classes = len(classes_set["source_classes"])
    model = CASEClassifier(num_classes)
    trainer = CASETrainer(
        model,
        num_source=num_classes,
        dataset_name=args.dataset,
        source_train_ds=source_train_ds,
        epochs=args.epochs,
        device=device,
        src_domain=source_domain,
        tgt_domain=target_domain,
        init_lr=args.lr,
        batch_size=args.batch_size,
        anchor_k=args.anchor_k,
    )

    h_scores = []
    for epoch in range(args.epochs):
        avg_loss, src_acc = trainer.train_epoch(source_train_dl, target_train_dl)
        result = trainer.inference(target_test_dl, class_list=classes_set["common_classes"])
        common_acc, unknown_acc, h_score = compute_h_score(
            result["per_class_correct"], result["per_class_num"], len(classes_set["common_classes"])
        )

        if epoch >= args.epochs - args.eval_last:
            h_scores.append(h_score)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | {source_domain}->{target_domain} | "
            f"loss={avg_loss:.4f} src_acc={src_acc * 100:.2f}% | "
            f"common={common_acc * 100:.2f}% unknown={unknown_acc * 100:.2f}% H={h_score * 100:.2f}%"
        )

    avg_h = sum(h_scores) / len(h_scores)
    print(f"Average H-score (last {args.eval_last} epochs): {avg_h * 100:.2f}%")
    return avg_h


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.source and args.target:
        pairs = [(args.source, args.target)]
    else:
        domains = DATASET_DOMAINS[args.dataset]
        pairs = [(s, t) for s in domains for t in domains if s != t]

    all_h = []
    for source_domain, target_domain in pairs:
        print("=" * 60)
        print(f"Task: {args.dataset} | {source_domain} -> {target_domain}")
        all_h.append(run_task(args, source_domain, target_domain, device))

    if len(all_h) > 1:
        print("=" * 60)
        print(f"Overall average H-score: {sum(all_h) / len(all_h) * 100:.2f}%")


if __name__ == "__main__":
    main()
