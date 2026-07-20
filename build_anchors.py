import argparse
import os

import clip
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet
from torch.utils.data import DataLoader

import data
from data import split_com_spvt_tpvt
from utils import DATASET_SPLITS, SIMPLE_IMAGENET_TEMPLATES, source_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Build semantic anchor vocabularies per domain")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Office", "OfficeHome", "VisDA", "DomainNet"])
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--clip-path", type=str, default="ckpt/clip/ViT-B-16.pt")
    parser.add_argument("--output-dir", type=str, default="labels_anchor")
    return parser.parse_args()


def get_wordnet_nouns():
    return sorted({
        synset.name().split(".")[0].replace("_", " ")
        for synset in wordnet.all_synsets(pos="n")
    })


def save_text_features(model, device, labels, save_dir, dataset_name):
    os.makedirs(save_dir, exist_ok=True)
    all_text_features = []

    with torch.no_grad():
        for label in labels:
            prompts = [template(label) for template in SIMPLE_IMAGENET_TEMPLATES]
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_features = F.normalize(model.encode_text(tokens), dim=-1)
            avg_feature = F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1)
            all_text_features.append(avg_feature)

    all_text_features = torch.cat(all_text_features, dim=0)
    torch.save(all_text_features.cpu(), os.path.join(save_dir, f"{dataset_name}.pt"))

    with open(os.path.join(save_dir, f"{dataset_name}_labels.txt"), "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")

    return all_text_features


def evaluate_topk(model, text_features, text_labels, dataloader, topk=2):
    model.eval()
    label_dict = {label.lower(): idx for idx, label in enumerate(text_labels)}
    text_features = text_features.to(torch.float32)

    all_topk_labels = set()
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = F.normalize(features, dim=-1)
            logits = 100.0 * (features @ text_features.T)
            targets = torch.tensor([label_dict[label.lower()] for label in labels])
            _, preds = logits.topk(topk, dim=1)

            for pred_row in preds.tolist():
                for idx in pred_row:
                    all_topk_labels.add(text_labels[idx])
            total += targets.size(0)

    return sorted(all_topk_labels)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_shared, n_source_private, n_total = DATASET_SPLITS[args.dataset]
    classes_set = split_com_spvt_tpvt(n_shared, n_source_private, n_total)

    model, _ = clip.load(args.clip_path, device=device)
    source_label = source_labels(args.dataset)
    src_label = [source_label[i] for i in classes_set["source_classes"]]
    combined_labels = sorted(set(src_label + get_wordnet_nouns()))

    feature_dir = "./representations/wordnet_labels"
    feature_path = os.path.join(feature_dir, f"{args.dataset}.pt")
    label_path = os.path.join(feature_dir, f"{args.dataset}_labels.txt")

    if not os.path.exists(feature_path):
        text_features = save_text_features(model, device, combined_labels, feature_dir, args.dataset)
    else:
        text_features = torch.load(feature_path, map_location="cpu")
        with open(label_path, encoding="utf-8") as f:
            combined_labels = [line.strip() for line in f]

    eval_dataset = data.UniDA_lastlayerfeature_withLabelName(args.dataset, args.domain)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)

    unique_labels = evaluate_topk(model, text_features, combined_labels, eval_loader, topk=args.topk)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.domain}_unique_top{args.topk}_labels.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_labels))

    print(f"Saved {len(unique_labels)} anchor labels to {output_path}")


if __name__ == "__main__":
    main()
