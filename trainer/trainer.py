import os
import torch
import torch.nn.functional as F
import itertools
from tqdm import tqdm
from torch import optim, nn
import time


class Trainer:
    def __init__(
        self,
        lr,
        lr_warmup_step,
        lr_milestones,
        lr_gamma,
        lambda_align,
        weight_decay,
        num_epochs,
        eval_step,
        early_stop,
        model,
        writer,
        logger,
        exp_dir,
        device,
    ):
        self.num_epochs = num_epochs
        self.eval_step = eval_step
        self.early_stop = early_stop
        self.model = model
        self.writer = writer
        self.logger = logger
        self.exp_dir = exp_dir
        self.best_save_dir = os.path.join(self.exp_dir, "best")
        self.device = device
        self.lambda_align = lambda_align

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = WarmupBeforeMultiStepLR(
            self.optimizer,
            warmup_step=lr_warmup_step,
            milestones=lr_milestones,
            gamma=lr_gamma,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, train_loader, valid_loader):
        best_val = 0.0
        patience = 0
        self.logger.info(f"Training starts - {self.num_epochs} epochs")

        for e in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            tqdm_train_loader = tqdm(train_loader)

            for *seq_info, next_item in tqdm_train_loader:
                seq_info = self.tensor_to_device(seq_info)
                next_item = self.tensor_to_device(next_item)

                self.optimizer.zero_grad()

                align_loss = self.calculate_modality_align_loss(next_item, train_loader.dataset.features)
                logits = self.model(seq_info)
                rec_loss = self.loss_fn(logits, next_item)
                loss = self.lambda_align * align_loss + rec_loss
                tqdm_train_loader.set_description(
                    f"Epoch {e} - Loss: {loss.item():.4f} CL Loss: {align_loss.item():.4f} Rec Loss: {rec_loss.item():.4f}"
                )

                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Epoch {e} - lr: {current_lr}, loss: {epoch_loss}")
            self.writer.add_scalar("Loss/train", epoch_loss, e)
            self.writer.add_scalar("LR", current_lr, e)
            self.scheduler.step()

            if e % self.eval_step == 0:
                with torch.no_grad():
                    best_val, patience = self.validate(e, valid_loader, best_val, patience)
                    if self.early_stop and patience * self.eval_step >= self.early_stop:
                        self.logger.info(f"Early stopping at epoch {e}")
                        break

        best_model_path = os.path.join(self.best_save_dir, "model.pt")
        self.model.load_state_dict(torch.load(best_model_path))

    def evaluate(self, eval_loader):
        self.model.eval()
        num_users = len(eval_loader.dataset)
        metrics_handler = MetricsHandler(num_users=num_users)
        total_inference_time = 0.0
        with torch.no_grad():
            for *seq_info, next_item in tqdm(eval_loader):
                seq_info = self.tensor_to_device(seq_info)
                start_time = time.perf_counter()
                scores = self.model(seq_info).detach().cpu()
                end_time = time.perf_counter()
                inference_time = end_time - start_time
                total_inference_time += inference_time
                metrics = calculate_metrics(scores, next_item, k_list=[1, 5, 10, 20])

                loss = self.loss_fn(scores, next_item)
                metrics["Loss"] = loss.item()
                metrics_handler.append_metrics(metrics)
        self.logger.info(f"Total inference time: {total_inference_time:.4f} seconds")
        return metrics_handler.get_metrics()

    def validate(self, epoch, valid_loader, best_val, patience):
        val_metrics = self.evaluate(valid_loader)
        val_log = ""
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"{k}/valid", v, epoch)
            val_log += f"{k}: {v:.5f} "
        self.logger.info(f"Epoch {epoch} validation - {val_log}")
        val = val_metrics["NDCG@20"]
        if val > best_val:
            best_val = val
            patience = 0
            self.logger.info("Save best model")
            save_model(self.best_save_dir, self.model.state_dict())
        else:
            patience += 1
        return (best_val, patience)

    def tensor_to_device(self, info):
        if type(info) == torch.Tensor:
            return info.to(self.device)
        return [[t.to(self.device) for t in elem] if type(elem) == list else elem.to(self.device) for elem in info]

    def load_model(self, saved_path):
        self.model.load_state_dict(torch.load(saved_path, map_location=torch.device(self.device)))

    def calculate_modality_align_loss(self, next_item, features):
        items = torch.argmax(next_item, dim=1)
        feats = [feat.to(self.device)[items] for feat in features]
        id_emb = self.model.id_emb_layer(items)
        f_embs = [f_emb_layer(f) for f_emb_layer, f in zip(self.model.f_emb_layers, feats)]
        return sum([contrastive_loss(id_emb, f_emb) for f_emb in f_embs])


class MetricsHandler:
    def __init__(self, num_users):
        self.metrics = None
        self.num_users = num_users

    def append_metrics(self, metrics):
        if not self.metrics:
            self.metrics = metrics
        else:
            for k, v in metrics.items():
                self.metrics[k] += v

    def get_metrics(self):
        return {k: ((v / self.num_users) if k != "Loss" else v) for k, v in self.metrics.items()}


class WarmupBeforeMultiStepLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_step=None, milestones=None, gamma=None, last_epoch=-1):
        self.gamma = 1

        def lr_lambda(step):
            if warmup_step and step < warmup_step:
                return (step + 1) / warmup_step
            if milestones and gamma and step + 1 in milestones:
                self.gamma *= gamma
            return self.gamma

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def calculate_metrics(scores, labels, k_list=[1, 5, 10, 20]):
    metrics = [f"{metric}@{k}" for metric, k in itertools.product(["NDCG", "HR"], k_list)]
    metrics_sum = {metric: 0.0 for metric in metrics}
    rank = (-scores).argsort(dim=1)
    labels_float = labels.float()

    max_k = max(k_list)
    cut = rank[:, :max_k]
    hits = labels_float.gather(dim=1, index=cut)
    del rank, labels_float, cut

    for k in k_list:
        hits_k = hits[:, :k]
        metrics_sum[f"HR@{k}"] = hits_k.sum().item()

        position = torch.arange(2, 2 + k).float()
        weights = 1 / torch.log2(position)

        dcg = (hits_k * weights).sum(dim=1)
        metrics_sum[f"NDCG@{k}"] = dcg.sum().item()  # for leave-one-out protocol, ndcg is same as dcg.

    return metrics_sum


def save_model(save_dir, model_state_dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_save_path = os.path.join(save_dir, "model.pt")
    torch.save(model_state_dict, model_save_path)


def contrastive_loss(x1, x2, temp=0.5):
    b = x1.shape[0]
    sim_matrix = F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=2) / temp
    labels = torch.ones(b).diag().to(sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
