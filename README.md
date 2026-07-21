# CASE: Cross-modal Semantic Anchoring Alignment and Structure Enhancement for Universal Domain Adaptation

Official PyTorch implementation of **CASE**, accepted by [IEEE Transactions on Multimedia (TMM)](https://github.com/yuefengw/CASE-UniDA).

## Overview

Universal Domain Adaptation (UniDA) transfers knowledge from a labeled source domain to an unlabeled target domain when label sets only partially overlap. CASE addresses this with two modules:

- **CSA (Cross-modal Semantic Anchoring Alignment):** builds a shared semantic anchor space and aligns cross-domain samples via JS-divergence-weighted contrastive learning.
- **SSE (Semantic Structure Enhancement):** aggregates anchor responses with K-Means clustering and feeds the enhanced representation to an All-in-One (AIO) classifier for unknown-class detection.

Backbone: frozen CLIP ViT-B/16 + lightweight MLP-Attention-MLP adapter (~3.15M trainable parameters).
[View the framework figure](./frame.pdf)
## Environment

```bash
conda create -n case python=3.10 -y
conda activate case
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
```

Download CLIP ViT-B/16 weights to `ckpt/clip/ViT-B-16.pt` (see [OpenAI CLIP](https://github.com/openai/CLIP)).

## Data Preparation

1. Download benchmark datasets and place images following the list files under `DataSets/`.
2. Extract CLIP features:

```bash
python get_features.py --dataset OfficeHome
python get_features.py --dataset Office
python get_features.py --dataset VisDA
python get_features.py --dataset DomainNet
```

3. Build semantic anchors (once per domain, or use the provided `labels_anchor/` files):

```bash
python build_anchors.py --dataset OfficeHome --domain Art --topk 2
```

## Training

Single task:

```bash
python train.py --dataset OfficeHome --source Art --target Clipart
```

Run all domain pairs:

```bash
python train.py --dataset OfficeHome
```

Main hyperparameters (defaults match the paper):

| Parameter | Default |
|-----------|---------|
| epochs | 80 |
| batch size | 36 |
| learning rate | 1e-3 |
| anchor top-k | 2 |
| evaluation | average H-score over last 10 epochs |

UniDA class splits: Office (10/10/11), OfficeHome (10/5/50), VisDA (6/6/6), DomainNet (150/50/145).

## Project Structure

```
CASE-UniDA/
├── train.py           # training & evaluation
├── model.py           # CASE (CSA + SSE + AIO)
├── data.py            # dataset loaders & class splits
├── utils.py           # labels, templates, helpers
├── get_features.py    # CLIP feature extraction
├── build_anchors.py   # semantic anchor construction
├── labels_anchor/     # pre-built anchor vocabularies
└── DataSets/          # dataset index files
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026case,
  title={Cross-modal Semantic Anchoring Alignment and Structure Enhancement for Universal Domain Adaptation},
  author={Li, Feijiang and Wang, Yuefeng and Qian, Yuhua and Wang, Jieting},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```

## License

This project is released for academic research use.
