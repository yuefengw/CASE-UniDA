import os

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset


class UniDA_dataset(Dataset):
    """Load raw images for CLIP feature extraction."""

    def __init__(self, dataset_name, domain_name, data_root="./DataSets"):
        txt_path = os.path.join(data_root, dataset_name, f"{domain_name}.txt")
        path_prefix = os.path.join(data_root, dataset_name)

        with open(txt_path, encoding="utf-8") as f:
            lst = []
            for line in f.readlines():
                impath = line.split(" ")[0]
                if dataset_name == "Office":
                    tmp = impath.split("/")
                    impath = os.path.join(path_prefix, tmp[0], tmp[2], tmp[3])
                elif dataset_name in ("OfficeHome", "myDataset"):
                    impath = os.path.join(path_prefix, impath[5:])
                elif dataset_name == "VisDA":
                    impath = os.path.join(path_prefix, domain_name, impath)
                elif dataset_name == "DomainNet":
                    impath = os.path.join(path_prefix, impath)
                label = int(line.split(" ")[1].strip())
                classname = impath.split("/")[-2].replace("_", " ")
                lst.append({"impath": impath, "label": label, "classname": classname})

        self.images = lst
        _, self.preprocess = clip.load("ckpt/clip/ViT-B-16.pt")

    def __getitem__(self, idx):
        item = self.images[idx % len(self.images)]
        img = self.preprocess(Image.open(item["impath"]))
        return img, item["label"]

    def __len__(self):
        return len(self.images)


class UniDA_lastlayerfeature(Dataset):
    """Load pre-extracted CLIP image features."""

    def __init__(self, dataset_name, domain_name, filter=None, data_root="./DataSets", feature_root="./representations"):
        self.dataset_name = dataset_name
        self.domain_name = domain_name
        self.filter = filter
        self.feature_root = feature_root

        txt_path = os.path.join(data_root, dataset_name, f"{domain_name}.txt")
        path_prefix = os.path.join(data_root, dataset_name)

        self.features = []
        self.labels = []

        with open(txt_path, encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue

                impath = line.split(" ")[0]
                label = int(line.split(" ")[1].strip())
                if self.filter is not None and not self.filter(label):
                    continue

                if dataset_name == "Office":
                    tmp = impath.split("/")
                    impath = os.path.join(path_prefix, tmp[0], tmp[2], tmp[3])
                elif dataset_name == "OfficeHome":
                    impath = os.path.join(path_prefix, impath[5:])
                elif dataset_name == "VisDA":
                    impath = os.path.join(path_prefix, domain_name, impath)
                elif dataset_name == "DomainNet":
                    impath = os.path.join(path_prefix, impath)

                classname = impath.split("/")[-2].replace("_", " ")
                self.features.append({"impath": impath, "label": label, "classname": classname})
                self.labels.append(label)

    def __getitem__(self, idx):
        feature_path = os.path.join(
            self.feature_root, self.dataset_name, self.domain_name, f"{idx}.pt"
        )
        feature = torch.load(feature_path, map_location="cpu").to(dtype=torch.float32)
        feature = torch.autograd.Variable(feature, requires_grad=False)
        return feature, self.features[idx]["label"]

    def __len__(self):
        return len(self.features)


class UniDA_lastlayerfeature_withLabelName(Dataset):
    """Load pre-extracted features with class names (for anchor construction)."""

    def __init__(self, dataset_name, domain_name, filter=None, data_root="./DataSets", feature_root="./representations"):
        self.dataset_name = dataset_name
        self.domain_name = domain_name
        self.filter = filter
        self.feature_root = feature_root

        txt_path = os.path.join(data_root, dataset_name, f"{domain_name}.txt")
        path_prefix = os.path.join(data_root, dataset_name)

        self.features = []
        self.labels = []

        with open(txt_path, encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue

                impath = line.split(" ")[0]
                label = int(line.split(" ")[1].strip())
                if self.filter is not None and not self.filter(label):
                    continue

                if dataset_name == "Office":
                    tmp = impath.split("/")
                    impath = os.path.join(path_prefix, tmp[0], tmp[2], tmp[3])
                elif dataset_name == "OfficeHome":
                    impath = os.path.join(path_prefix, impath[5:])
                elif dataset_name == "VisDA":
                    impath = os.path.join(path_prefix, domain_name, impath)
                elif dataset_name == "DomainNet":
                    impath = os.path.join(path_prefix, impath)

                classname = impath.split("/")[-2].replace("_", " ")
                self.features.append({"impath": impath, "label": label, "classname": classname})
                self.labels.append(label)

    def __getitem__(self, idx):
        feature_path = os.path.join(
            self.feature_root, self.dataset_name, self.domain_name, f"{idx}.pt"
        )
        feature = torch.load(feature_path, map_location="cpu").to(dtype=torch.float32)
        feature = torch.autograd.Variable(feature, requires_grad=False)
        return feature, self.features[idx]["classname"]

    def __len__(self):
        return len(self.features)


def split_com_spvt_tpvt(n_shared, n_source_private, n_total):
    """Split classes into common, source-private, and target-private sets."""
    a, b = n_shared, n_source_private
    c = n_total - a - b

    common_classes = list(range(a))
    source_private_classes = [i + a for i in range(b)]
    target_private_classes = [i + a + b for i in range(c)]

    source_classes = common_classes + source_private_classes
    target_classes = common_classes + target_private_classes

    tp_classes = sorted(set(target_classes) - set(source_classes))
    sp_classes = sorted(set(source_classes) - set(target_classes))
    common_classes = sorted(set(source_classes) - set(sp_classes))

    return {
        "source_classes": source_classes,
        "target_classes": target_classes,
        "tp_classes": tp_classes,
        "sp_classes": sp_classes,
        "common_classes": common_classes,
    }
