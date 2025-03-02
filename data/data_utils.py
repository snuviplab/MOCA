import re
import pandas as pd
import json
import requests
import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from tqdm import tqdm
import ast


def get_col_data(data, col, dtype):
    try:
        if dtype == "str":
            return re.search(rf"'{col}':[ ]*'([^']+)'", data)[1]
        elif dtype == "list":
            return ast.literal_eval("[" + re.search(rf"'{col}':[ ]*\[+([^\]]+)\]+", data)[1] + "]")
    except:
        return ""


def read_meta(datapath):
    with open(datapath, "r") as f:
        items = []
        notitle = 0
        nodesc = 0
        for line in tqdm(f.readlines()):
            items.append(
                {
                    "asin": get_col_data(line, "asin", "str"),
                    "title": get_col_data(line, "title", "str"),
                    "description": get_col_data(line, "description", "str"),
                    "price": get_col_data(line, "price", "str"),
                    "imUrl": get_col_data(line, "imUrl", "str"),
                    "related": get_col_data(line, "related", "str"),
                    "salesRank": get_col_data(line, "salesRank", "str"),
                    "brand": get_col_data(line, "brand", "str"),
                    "categories": get_col_data(line, "categories", "list"),
                }
            )
        items = pd.DataFrame(items)
    return items


def read_reviews(datapath):
    with open(datapath, "r") as f:
        reviews = f.readlines()
    reviews = pd.DataFrame([json.loads(review) for review in reviews])
    return reviews


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img).unsqueeze(0)  # Add batch dimension


def extract_image_features(model, chunk, chunk_idx):
    features = {}
    for url, asin in tqdm(
        zip(chunk["imUrl"], chunk["asin"]), total=len(chunk), desc=f"Extracting image features for chunk {chunk_idx}"
    ):
        try:
            img = download_image(url)
            img_tensor = preprocess_image(img).to(model.device)
            with torch.no_grad():
                output = model(img_tensor)
                features[asin] = output.last_hidden_state.detach().cpu()
        except Exception as e:
            print(f"Extracting image features from {asin} failed!")
            print(e)
    return features


def extract_text_features(chunk, chunk_idx, tokenizer, model):
    features = {}
    for text, asin in tqdm(
        zip(chunk["text"], chunk["asin"]), total=len(chunk), desc=f"Extracting text features for chunk {chunk_idx}"
    ):
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                features[asin] = outputs.pooler_output.detach().cpu()
        except Exception as e:
            print(f"Extracting text features from {asin} failed!")
            print(e)
    return features
