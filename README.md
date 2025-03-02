# Mixture of Conditional Attention for Multimodal Fusion in Sequential Recommendation (MOCA)

This is the official PyTorch implementation of the paper **"Mixture of Conditional Attention for Multimodal Fusion in Sequential Recommendation"** accepted to PAKDD 2025.

## Paper Abstract
Sequential Recommender (SR) systems stand out for their ability to capture dynamic user preference, and multimodal side information has been incorporated to improve the recommendation quality. Most existing approaches, however, rely on predefined deterministic rules reflecting some inductive biases. Despite their useful guidance for the training process, they also limit the model’s capability to explore cross-modal relationships. To address this problem, we introduce the Mixture of Conditional Attention (MOCA) framework, which learns diverse and flexible attention patterns directly from data. MOCA utilizes 1) a conditional attention mechanism to focus on the most relevant features aligned with user intent, and 2) a mixture-of-experts approach to capture a wide range of user preferences effectively. Extensive experiments on multiple datasets demonstrate the superiority of our model over state-of-the-art SR models.

<p align="center"><img src=".github/MOCA_overall.png" height="300"/></p>

## Usage 

### Environment Setup
```bash
# Create and activate conda environment
conda env create -n moca python=3.10
conda activate moca

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Train MOCA model on Beauty dataset
python main.py \
    --mode train \
    --dataset beauty \
    --exp_name moca_beauty
```

### Testing
```bash
# Evaluate pre-trained model
python main.py \
    --mode test \
    --dataset beauty \
    --saved_model_path experiments/beauty/moca_beauty/best/model.pt
```

For detailed configuration options and available arguments, please refer to `utils.py`.