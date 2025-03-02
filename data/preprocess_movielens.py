import sys
import argparse
import pandas as pd
import numpy as np
from transformers import ViTModel, BertTokenizer, BertModel
import os
import glob
import random
import pickle
import torch
from PIL import Image
import concurrent.futures
from data_utils import extract_text_features, preprocess_image


def main(in_dir, out_dir, num_workers):
    print("Read ratings data")
    ratings = pd.read_csv(
        os.path.join(in_dir, "ratings.dat"), sep="::", names=["user_id", "movie_id", "rating", "timestamp"]
    )
    print("Read text data")
    texts = pd.read_csv(os.path.join(in_dir, "movies_text.csv"))
    texts = {
        int(movie_id): f"title: {title}. overview: {overview}"
        for movie_id, title, overview in zip(texts["movieId"], texts["title"], texts["overview"])
    }

    img_feat_path = os.path.join(out_dir, f"image_ml-1m.pkl")
    if os.path.exists(img_feat_path):
        print(f"Load image feature from {img_feat_path}")
        with open(img_feat_path, "rb") as pf:
            img_feat = pickle.load(pf)
    else:
        print("Extract image feature")
        img_feat = {}
        model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(extract_image_features, os.path.join(in_dir, f"frames/{item_id}"), model)
                for item_id in sorted(ratings["movie_id"].unique())
            ]
            for future in concurrent.futures.as_completed(futures):
                feature, item_id = future.result()
                img_feat[item_id] = feature
        print("Save image feature")
        with open(img_feat_path, "wb") as pf:
            pickle.dump(img_feat, pf)

    text_feat_path = os.path.join(out_dir, f"text_ml-1m.pkl")
    if os.path.exists(text_feat_path):
        print(f"Load text feature from {text_feat_path}")
        with open(text_feat_path, "rb") as pf:
            text_feat = pickle.load(pf)
    else:
        print("Extract text feature")
        text_feat = {}
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(extract_text_features, text, item_id, tokenizer, model)
                for item_id, text in texts.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                feature, item_id = future.result()
                text_feat[item_id] = feature
        print("Save text feature")
        with open(text_feat_path, "wb") as pf:
            pickle.dump(text_feat, pf)
    img_feat = {k: v for k, v in img_feat.items() if v is not None}
    text_feat = {k: v for k, v in text_feat.items() if v is not None}

    print("Filter ratings")
    filtered_items = set(img_feat.keys()).intersection(set(text_feat.keys()))
    img_feat = {k: v for k, v in img_feat.items() if k in filtered_items}
    text_feat = {k: v for k, v in text_feat.items() if k in filtered_items}
    item_cnt = ratings.groupby("movie_id").count()["user_id"].reset_index()
    filtered_item = item_cnt[item_cnt["user_id"] >= 5]["movie_id"]
    ratings = ratings[ratings["movie_id"].isin(filtered_item)]
    user_cnt = ratings.groupby("user_id").count()["movie_id"].reset_index()
    filtered_user = user_cnt[user_cnt["movie_id"] >= 5]["user_id"]
    ratings = ratings[ratings["user_id"].isin(filtered_user)]

    print("Make mapper")
    unique_items = ratings["movie_id"].unique()
    item_to_id = dict(zip(unique_items, range(1, len(unique_items) + 1)))  # starting from 1
    id_to_item = {v: k for k, v in item_to_id.items()}
    with open(os.path.join(out_dir, f"id_to_item_ml-1m.pkl"), "wb") as pf:
        pickle.dump(id_to_item, pf)

    print("Make inter data")
    ratings["item_id"] = [item_to_id[movie_id] for movie_id in ratings["movie_id"]]
    seq_all = ratings.groupby("user_id")["item_id"].apply(list).reset_index()
    seq_train = [seq[:-2] for seq in seq_all["item_id"]]
    min_seq_len = 1

    split_seq = lambda seq: [(tuple(seq[:i]), tuple(seq[i:])) for i in range(min_seq_len, len(seq))]
    seq_train = [(subseq, labels) for seq in seq_train for subseq, labels in split_seq(seq)]
    seq_train = pd.DataFrame([{"seq": seq, "label": labels[0], "labels": labels} for seq, labels in seq_train])
    seq_train.drop_duplicates(inplace=True)
    seq_valid = pd.DataFrame([{"seq": seq[:-2], "label": seq[-2]} for seq in seq_all["item_id"]])
    seq_test = pd.DataFrame([{"seq": seq[:-1], "label": seq[-1]} for seq in seq_all["item_id"]])
    seq_train.to_parquet(os.path.join(out_dir, f"train_ml-1m.parquet"))
    seq_valid.to_parquet(os.path.join(out_dir, f"valid_ml-1m.parquet"))
    seq_test.to_parquet(os.path.join(out_dir, f"test_ml-1m.parquet"))

    print("Make meta data")
    num_items = len(item_to_id)
    img_dim = list(img_feat.values())[0].shape[0]
    text_dim = list(text_feat.values())[0][0, :].shape[0]
    img_feat_matrix = np.zeros((num_items + 1, img_dim))  # zero index is for dummy (padding)
    for item_id in range(1, len(item_to_id) + 1):
        img_feat_matrix[item_id] = img_feat[id_to_item[item_id]]
    text_feat_matrix = np.zeros((num_items + 1, text_dim))  # zero index is for dummy (padding)
    for item_id in range(1, len(item_to_id) + 1):
        text_feat_matrix[item_id] = text_feat[id_to_item[item_id]][0, :]
    np.save(os.path.join(out_dir, f"image_ml-1m.npy"), img_feat_matrix)
    np.save(os.path.join(out_dir, f"text_ml-1m.npy"), text_feat_matrix)

    print("Preprocessing done")
    print(f"# of users: {len(seq_all)}")
    print(f"# of sequences (train): {len(seq_train)}")
    print(f"# of items: {num_items}")
    print(f"# of interactions: {len(ratings)}")
    print(f"Length of sequences:")
    print(pd.DataFrame([len(seq) for seq in seq_all["item_id"]]).describe())


def extract_image_features(frame_dir, model, sample_frames=64):
    item_id = int(os.path.basename(frame_dir))
    print(f"Extracting image features from {item_id} starts")
    try:
        img_files = glob.glob(os.path.join(frame_dir, "*.jpg"))
        sampled = set(random.sample(range(int(len(img_files) * 0.1), int(len(img_files) * (0.9))), sample_frames))
        img_tensor = torch.cat(
            [
                preprocess_image(Image.open(img_file))
                for frame_no, img_file in enumerate(img_files)
                if frame_no in sampled
            ]
        )
        print(f"Make image tensor done: {item_id}")
        output = model.to("cuda")(img_tensor.to("cuda"))
        feature = output.last_hidden_state.mean(axis=0)[0, :].detach().cpu()
    except Exception as e:
        print(f"Failed: {item_id}")
        print(e)
        feature = None
    else:
        print(f"Success: {item_id} (shape: {feature.shape})")
    return (feature, item_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    data_dir = "/data/dataset/movielens/ml-1m"
    out_dir = os.path.join("processed", "ml-1m")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sys.exit(main(data_dir, out_dir, args.num_workers))
