## Dataset Preparation

### 1. Download Raw Data

#### Amazon Datasets
The Amazon datasets are from [McAuley et al., SIGIR 2015](http://jmcauley.ucsd.edu/data/amazon/). Download the following files:
- Reviews: `reviews_{DATASET}_5.json`
- Metadata: `meta_{DATASET}.json`

Place the files in the following structure:
```
raw/
└── amazon/
├── reviews/
│ └── reviews_{DATASET}5.json
└── meta/
  └── meta_{DATASET}.json
```

#### MovieLens-1M Dataset
The MovieLens-1M dataset is from [Harper & Konstan, 2015](https://grouplens.org/datasets/movielens/1m/), with additional content information collected by us.

### 2. Preprocess Data

Run the preprocessing scripts with the following commands:

For example, to preprocess the Amazon Beauty dataset:

```bash
python preprocess_amazon.py --dataset Beauty --output_dir processed/amazon/Beauty
```



## Dataset statistics after preprocessing

| Dataset      | \# Users | \# Items | \# Actions | Avg. A./U. | Max. A./U. |
|--------------|-----------|-----------|------------|------------|------------|
| Beauty       | 19,766    | 11,168    | 173,608    | 8.78       | 191        |
| Sports       | 30.902                                        | 16,678                                        | 255,853                                         | 8.28                                            | 277                                             |
| Toys         | 16,968                                        | 11,016                                        | 144,372                                         | 8.51                                            | 519                                             |
| Cloth        | 8,845                                         | 10,342                                        | 59,486                                          | 6.73                                            | 64                                              |
| Movielens-1M | 6,040                                         | 3,416                                         | 999,611                                         | 165.50                                          | 2277                                            |
