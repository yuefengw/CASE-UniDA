import math

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.autograd import Variable

class SimpleAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=max(1, dim // 64),
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(attn_output + x)
        return x.squeeze(1)


class CASEClassifier(nn.Module):
    """All-in-One classifier with MLP-Attention-MLP adaptation network."""

    def __init__(self, num_classes, input_size=1024, temp=0.05, norm=True):
        super().__init__()
        self.num_classes = num_classes
        self.norm = norm
        self.temp = temp

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(inplace=True),
            SimpleAttentionBlock(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, input_size),
            nn.GELU(),
        )
        self.fc1 = nn.Linear(input_size, input_size, bias=False)
        self.fc2 = nn.Linear(input_size, 2 * num_classes, bias=False)
        self.rulu1 = nn.ReLU()

    def forward(self, x, return_feat=False):
        x = self.feature_extractor(x)
        if return_feat:
            return x

        if self.norm:
            x = F.normalize(x)
            x = self.fc1(x) / self.temp
            x = self.fc2(self.rulu1(x))
        else:
            x = self.fc1(x) / self.temp
            x = self.fc2(torch.sigmoid(x))
        return x

    def weight_norm(self):
        w = self.fc1.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc1.weight.data = w.div(norm.expand_as(w))

    def get_features(self, x):
        return self.forward(x, return_feat=True)


class CASETrainer:
    """CASE framework: CSA (Cross-modal Semantic Anchoring) + SSE (Semantic Structure Enhancement)."""

    def __init__(
        self,
        model,
        num_source,
        dataset_name,
        source_train_ds,
        epochs,
        device,
        src_domain,
        tgt_domain,
        init_lr=1e-3,
        batch_size=36,
        anchor_k=2,
        clip_path="ckpt/clip/ViT-B-16.pt",
    ):
        self.device = device
        self.model = model.to(device)
        self.dataset_name = dataset_name
        self.Cs = num_source
        self.anchor_k = anchor_k

        self.clip_model, _ = clip.load(clip_path, device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        total_iterations = max(1, epochs * len(source_train_ds) // batch_size)
        alpha, beta = 0.1, 0.75

        def lr_lambda(step):
            return (1 + alpha * step / total_iterations) ** (-beta)

        self._load_shared_embeddings(src_domain, tgt_domain)

        self.optimizer = optim.SGD(self.model.parameters(), lr=init_lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.ce_loss = nn.CrossEntropyLoss()
        self.model.weight_norm()

    def _load_shared_embeddings(self, src_domain, tgt_domain):
        k = self.anchor_k
        with open(f"labels_anchor/{src_domain}_unique_top{k}_labels.txt", encoding="utf-8") as f:
            src_words = [line.strip() for line in f if line.strip()]
        with open(f"labels_anchor/{tgt_domain}_unique_top{k}_labels.txt", encoding="utf-8") as f:
            tgt_words = [line.strip() for line in f if line.strip()]

        shared_words = sorted(set(src_words) & set(tgt_words))
        print(f"Found {len(shared_words)} shared semantic anchors")

        text_inputs = clip.tokenize(shared_words).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.W = text_features.to(torch.float32)
        self.m = len(shared_words)

    def _project_to_W(self, features, temperature=0.1):
        similarity = features @ self.W.T
        return F.softmax(similarity / temperature, dim=-1)

    def _compute_js_divergence(self, probs):
        pi = probs.unsqueeze(1)
        pj = probs.unsqueeze(0)
        m = 0.5 * (pi + pj)
        kl1 = pi * (torch.log(pi + 1e-8) - torch.log(m + 1e-8))
        kl2 = pj * (torch.log(pj + 1e-8) - torch.log(m + 1e-8))
        js = 0.5 * (kl1.sum(-1) + kl2.sum(-1))
        return 1 - js / math.log(2)

    def _compute_contrastive_loss(self, features, p_ij, tau=0.1):
        norm_features = F.normalize(features, p=2, dim=1)
        logits = norm_features @ norm_features.T / tau
        mask = torch.eye(features.size(0), dtype=torch.bool, device=self.device)
        logits = logits.masked_fill(mask, -1e9)
        q_ij = torch.sigmoid(logits)
        loss = -(p_ij * torch.log(q_ij + 1e-8) + (1 - p_ij) * torch.log(1 - q_ij + 1e-8))
        return loss.mean()

    def _ova_loss_amlp(self, out_open, label, lambada=1.5):
        label_p = torch.zeros((out_open.size(0), out_open.size(2)), dtype=torch.long, device=out_open.device)
        label_range = torch.arange(0, out_open.size(0), device=out_open.device)
        label_p[label_range, label] = 1
        label_n = 1 - label_p

        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :] + 1e-8) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] + 1e-8) * label_n, 1)[0])
        known_p_first = torch.max(out_open[:, 1, :], dim=1)[0]
        unknown_p_first = torch.max(out_open[:, 0, :], dim=1)[0]
        loss_connection_first = torch.mean(-torch.log(torch.sigmoid(known_p_first - unknown_p_first))) * lambada
        return open_loss_pos, open_loss_neg, loss_connection_first

    def reduce_dim_by_column_kmeans(self, proj_tensor):
        """SSE: aggregate anchor responses into l cluster dimensions (l <= 512)."""
        target_dim = 512
        batch_size, num_anchors = proj_tensor.size()

        if num_anchors == target_dim:
            return proj_tensor
        if num_anchors < target_dim:
            return F.pad(proj_tensor, (0, target_dim - num_anchors), mode="constant", value=0)

        proj_tensor_t = proj_tensor.T.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=target_dim, n_init="auto").fit(proj_tensor_t)
        labels = kmeans.labels_

        reduced_features = []
        proj_np = proj_tensor.cpu().detach().numpy()
        for cluster_id in range(target_dim):
            indices = labels == cluster_id
            if indices.sum() == 0:
                reduced_features.append(np.zeros((batch_size,)))
            else:
                reduced_features.append(proj_np[:, indices].mean(axis=1))

        reduced_tensor = np.stack(reduced_features, axis=1)
        return torch.tensor(reduced_tensor, dtype=proj_tensor.dtype, device=proj_tensor.device)

    def _augment_with_sse(self, visual_features, anchor_probs):
        semantic_structure = self.reduce_dim_by_column_kmeans(anchor_probs)
        return torch.cat([visual_features, semantic_structure], dim=1)

    def train_epoch(self, source_loader, target_loader):
        self.model.train()
        total_loss = 0.0
        src_correct, src_total = 0, 0

        for batch_idx, ((src_data, src_labels), (tgt_data, _)) in enumerate(zip(source_loader, target_loader)):
            src_data = Variable(src_data).to(self.device)
            tgt_data = Variable(tgt_data).to(self.device)
            src_labels = Variable(src_labels).to(self.device)

            mixed_features = torch.cat([src_data, tgt_data])
            proj_probs = self._project_to_W(mixed_features)
            mixed_aug = self._augment_with_sse(mixed_features, proj_probs)
            mixed_features_proj = self.model.get_features(mixed_aug)
            p_ij = self._compute_js_divergence(proj_probs)
            loss_csa = self._compute_contrastive_loss(mixed_features_proj, p_ij)

            proj_src = proj_probs[: src_data.size(0)]
            src_aug = self._augment_with_sse(src_data, proj_src)
            out = self.model(src_aug)

            out_s = out.reshape(out.shape[0], 2, -1)[:, 1, :]
            loss_s = self.ce_loss(out_s, src_labels)

            out_open = F.softmax(out.reshape(out.shape[0], -1), dim=1).view(out.shape[0], 2, -1)
            open_loss_pos, open_loss_neg, loss_connection = self._ova_loss_amlp(out_open, src_labels)
            loss_open = (open_loss_pos + open_loss_neg + loss_connection) / 3

            loss = loss_open + loss_s + loss_csa

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            _, predicted = out_s.max(1)
            src_total += src_labels.size(0)
            src_correct += predicted.eq(src_labels).sum().item()

        avg_loss = total_loss / (batch_idx + 1)
        acc = src_correct / src_total if src_total > 0 else 0.0
        return avg_loss, acc

    @torch.no_grad()
    def inference(self, dataloader, class_list):
        self.model.eval()
        device = next(self.model.parameters()).device

        per_class_num = np.zeros(len(class_list) + 1)
        per_class_correct = np.zeros(len(class_list) + 1, dtype=np.float32)

        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            proj_probs = self._project_to_W(data)
            data_aug = self._augment_with_sse(data, proj_probs)

            mask = ~torch.isin(labels, torch.tensor(class_list, device=device))
            labels[mask] = len(class_list)

            out = self.model(data_aug)
            out_positive = out.view(out.size(0), 2, -1)[:, 1, :]
            preds = out_positive.argmax(1)

            out = F.softmax(out, dim=1).view(out.size(0), 2, -1)
            out_np = out.cpu().numpy()
            mask_unk = np.where(
                np.max(out_np[:, 0, :], axis=1) >= np.max(out_np[:, 1, :], axis=1)
            )[0]
            preds[mask_unk] = len(class_list)

            for label, pred in zip(labels, preds):
                per_class_num[label] += 1
                if label == pred:
                    per_class_correct[label] += 1

        return {"per_class_correct": per_class_correct, "per_class_num": per_class_num}
