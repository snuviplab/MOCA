import sys
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import ViTModel, BertTokenizer, BertModel
import os
import pickle
import concurrent.futures
from data_utils import (
    read_meta,
    read_reviews,
    extract_image_features,
    extract_text_features,
)


def main(dataset, review_dir, meta_dir, out_dir, num_workers):
    print("Read review data")
    review = read_reviews(os.path.join(review_dir, f"reviews_{dataset}_5.json"))
    print("Read meta data")
    meta = read_meta(os.path.join(meta_dir, f"meta_{dataset}.json"))

    print("Filter meta data")
    items_in_reviews = set(review["asin"].unique())
    meta = meta[meta["asin"].isin(items_in_reviews)]
    meta = meta[meta["title"] != ""]
    meta = meta[meta["imUrl"].str.startswith("http")]
    meta["text"] = [title + " " + desc for title, desc in zip(meta["title"], meta["description"])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)

    img_feat_path = os.path.join(out_dir, f"image_{dataset}.pkl")
    if os.path.exists(img_feat_path):
        print(f"Load image feature from {img_feat_path}")
        with open(img_feat_path, "rb") as pf:
            img_feat = pickle.load(pf)
    else:
        print("Extract image feature")
        img_feat = {}
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        model.to(device)

        chunk_size = len(meta) // num_workers
        chunks = [meta[["imUrl", "asin"]].iloc[i : i + chunk_size] for i in range(0, len(meta), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(extract_image_features, model, chunk, chunk_idx)
                for chunk_idx, chunk in enumerate(chunks)
            ]
            for future in concurrent.futures.as_completed(futures):
                features = future.result()
                img_feat.update(features)
        print("Save image feature")
        with open(img_feat_path, "wb") as pf:
            pickle.dump(img_feat, pf)

    text_feat_path = os.path.join(out_dir, f"text_{dataset}.pkl")
    if os.path.exists(text_feat_path):
        print(f"Load text feature from {text_feat_path}")
        with open(text_feat_path, "rb") as pf:
            text_feat = pickle.load(pf)
    else:
        print("Extract text feature")
        text_feat = {}
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.to(device)

        chunk_size = len(meta) // num_workers
        chunks = [meta[["text", "asin"]].iloc[i : i + chunk_size] for i in range(0, len(meta), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(extract_text_features, chunk, chunk_idx, tokenizer, model)
                for chunk_idx, chunk in enumerate(chunks)
            ]
            for future in concurrent.futures.as_completed(futures):
                features = future.result()
                text_feat.update(features)
        print("Save text feature")
        with open(text_feat_path, "wb") as pf:
            pickle.dump(text_feat, pf)
    img_feat = {k: v for k, v in img_feat.items() if v is not None}
    text_feat = {k: v for k, v in text_feat.items() if v is not None}

    print("Filter reviews")
    filtered_items = set(img_feat.keys()).intersection(set(text_feat.keys()))
    img_feat = {k: v for k, v in img_feat.items() if k in filtered_items}
    text_feat = {k: v for k, v in text_feat.items() if k in filtered_items}
    review = review[review["asin"].isin(filtered_items)]
    seqlen = review.groupby("reviewerID").count()["asin"].reset_index()
    filtered_user = seqlen[seqlen["asin"] >= 5]["reviewerID"]
    review = review[review["reviewerID"].isin(filtered_user)]

    print("Make mapper")
    unique_items = review["asin"].unique()
    item_to_id = dict(zip(unique_items, range(1, len(unique_items) + 1)))  # starting from 1
    id_to_item = {v: k for k, v in item_to_id.items()}
    with open(os.path.join(out_dir, f"id_to_item_{dataset}.pkl"), "wb") as pf:
        pickle.dump(id_to_item, pf)

    print("Make inter data")
    review["item_id"] = [item_to_id[asin] for asin in review["asin"]]
    seq_all = review.groupby("reviewerID")["item_id"].apply(list).reset_index()
    seq_train = [seq[:-2] for seq in seq_all["item_id"]]
    min_seq_len = 1

    split_seq = lambda seq: [(tuple(seq[:i]), tuple(seq[i:])) for i in range(min_seq_len, len(seq))]
    seq_train = [(subseq, labels) for seq in seq_train for subseq, labels in split_seq(seq)]
    seq_train = pd.DataFrame([{"seq": seq, "label": labels[0], "labels": labels} for seq, labels in seq_train])
    seq_train.drop_duplicates(inplace=True)

    seq_valid = pd.DataFrame([{"seq": seq[:-2], "label": seq[-2]} for seq in seq_all["item_id"]])
    seq_test = pd.DataFrame([{"seq": seq[:-1], "label": seq[-1]} for seq in seq_all["item_id"]])
    seq_train.to_parquet(os.path.join(out_dir, f"train_{dataset}.parquet"))
    seq_valid.to_parquet(os.path.join(out_dir, f"valid_{dataset}.parquet"))
    seq_test.to_parquet(os.path.join(out_dir, f"test_{dataset}.parquet"))

    print("Make meta data")
    num_items = len(item_to_id)
    img_dim = list(img_feat.values())[0][0, 0, :].shape[0]
    text_dim = list(text_feat.values())[0][0, :].shape[0]
    img_feat_matrix = np.zeros((num_items + 1, img_dim))  # zero index is for dummy (padding)
    for item_id in range(1, len(item_to_id) + 1):
        img_feat_matrix[item_id] = img_feat[id_to_item[item_id]][0, 0, :]
    text_feat_matrix = np.zeros((num_items + 1, text_dim))  # zero index is for dummy (padding)
    for item_id in range(1, len(item_to_id) + 1):
        text_feat_matrix[item_id] = text_feat[id_to_item[item_id]][0, :]
    np.save(os.path.join(out_dir, f"image_{dataset}.npy"), img_feat_matrix)
    np.save(os.path.join(out_dir, f"text_{dataset}.npy"), text_feat_matrix)

    print("Preprocessing done")
    print(f"# of users: {len(seq_all)}")
    print(f"# of sequences (train): {len(seq_train)}")
    print(f"# of items: {num_items}")
    print(f"# of interactions: {len(review)}")
    print(f"Length of sequences:")
    print(pd.DataFrame([len(seq) for seq in seq_all["item_id"]]).describe())


if __name__ == "__main__":
    dataset_list = [
        "Beauty",
        "Clothing_Shoes_and_Jewelry",
        "Sports_and_Outdoors",
        "Toys_and_Games",
        "Cell_Phones_and_Accessories",
        "Home_and_Kitchen",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, choices=dataset_list)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if not args.dataset:
        print(f"Give dataset as an argument: {dataset_list}")
    else:
        review_dir = "raw/amazon/reviews"
        meta_dir = "raw/amazon/meta"
        out_dir = os.path.join("processed", args.dataset)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        sys.exit(main(args.dataset, review_dir, meta_dir, out_dir, args.num_workers))
