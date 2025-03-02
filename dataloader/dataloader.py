import numpy as np
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm


class SequenceDataset:
    def __init__(self, df, num_items, maxlen, features, label_col="label", gamma=0.5):
        self.df = df
        self.num_items = num_items
        self.maxlen = maxlen
        if label_col == "labels":
            max_num_labels = max([len(labels) for labels in df["labels"]])
            self.gamma = torch.pow(gamma, torch.arange(max_num_labels))

        self.seqs = self.df["seq"]
        self.label = self.df[label_col]
        self.features = features

    def __len__(self):
        return len(self.df)

    def make_fixed_length(self, seq):
        seq = torch.tensor(seq)
        seqlen = len(seq)
        if seqlen > self.maxlen:
            seq = seq[-self.maxlen :]
        else:
            padded = torch.zeros(self.maxlen - seqlen)
            seq = torch.cat([padded, seq], axis=0)
        return seq

    def make_label(self, label):
        label = one_hot(
            torch.tensor(label),
            num_classes=self.num_items + 1,
        ).float()
        if label.ndim == 2:
            label = (label * self.gamma[: label.shape[0]].unsqueeze(1)).sum(axis=0)
        return label

    def __getitem__(self, idx):
        seq = self.make_fixed_length(self.seqs.iloc[idx]).long()
        label = self.make_label(self.label.iloc[idx])
        seq_feats = [feature[seq] for feature in self.features]
        return (seq, *seq_feats, label)
