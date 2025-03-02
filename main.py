import os
import sys
from torch.utils.data import DataLoader
from utils import parse_arguments, get_logger, get_writer, set_seed
from data import get_id_to_item, get_inter_data, get_feature
from dataloader import SequenceDataset
from model import MOCA
from trainer import Trainer


def main():
    args = parse_arguments()
    exp_dir = os.path.join(os.path.join(args.exp_dir, args.dataset.lower()), args.exp_name)
    logger = get_logger(exp_dir)
    logger.info(args)
    set_seed(args.seed, logger)

    logger.info("Load Data")
    data_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(data_dir):
        logger.error(f"{data_dir} does not exist. Preprocess the data first.")
        return 1
    id_to_item = get_id_to_item(data_dir, args.dataset)
    num_items = len(id_to_item)
    train, valid, test = get_inter_data(data_dir, args.dataset)
    features = get_feature(data_dir, args.dataset, args.feature_cols)
    logger.info(f"# of items: {num_items}, # of seqs: {len(train)}(train), {len(test)}(eval)")

    logger.info("Prepare Data")
    features = get_feature(data_dir, args.dataset, args.feature_cols)
    feature_dims = [feat.shape[1] for feat in features]
    train_dataset = SequenceDataset(
        train,
        num_items,
        args.maxlen,
        features,
        label_col="labels" if args.label_gamma else "label",
        gamma=args.label_gamma,
    )
    valid_dataset = SequenceDataset(valid, num_items, args.maxlen, features)
    test_dataset = SequenceDataset(test, num_items, args.maxlen, features)
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=args.num_workers
    )
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    logger.info("Load Model")
    model = MOCA(
        dim=args.dim,
        num_feats=len(features) + 1,  # including ID
        feature_dims=feature_dims,
        num_items=num_items,
        maxlen=args.maxlen,
        num_layers=args.num_layers,
        dim_head=args.dim_head,
        num_heads=args.num_heads,
        feature_transformer=args.feature_transformer,
        feature_num_routed_queries=args.feature_num_routed_queries,
        feature_num_routed_key_values=args.feature_num_routed_key_values,
        feature_num_experts=args.feature_num_experts,
        item_transformer=args.item_transformer,
        item_num_routed_queries=args.item_num_routed_queries,
        item_num_routed_key_values=args.item_num_routed_key_values,
        item_num_experts=args.item_num_experts,
        feature_dropout=args.feature_dropout,
        attn_dropout=args.attn_dropout,
        ff_dropout=args.ff_dropout,
        use_flash=args.use_flash,
        null_token_to_unrouted=args.null_token_to_unrouted,
    )
    model = model.to(args.device)

    writer = get_writer(args, exp_dir)
    trainer = Trainer(
        lr=args.lr,
        lr_warmup_step=args.lr_warmup_step,
        lr_milestones=args.lr_milestones,
        lr_gamma=args.lr_gamma,
        lambda_align=args.lambda_align,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        eval_step=args.eval_step,
        early_stop=args.early_stop,
        model=model,
        writer=writer,
        logger=logger,
        exp_dir=exp_dir,
        device=args.device,
    )
    if args.mode == "train":
        logger.info("Train starts")
        trainer.train(train_loader, valid_loader)
    else:
        if not args.saved_model_path:
            logger.error("For evaluation mode, saved model path must be given.")
            return 1
        trainer.load_model(args.saved_model_path)

    logger.info("Evaluation starts")
    test_metrics = trainer.evaluate(test_loader)
    test_log = ""
    for k, v in test_metrics.items():
        test_log += f"{k}: {v:.5f} "
    logger.info(f"Test - {test_log}")
    if args.mode == "train":
        writer.add_hparams(hparam_dict={"exp_dir": exp_dir}, metric_dict=test_metrics)
        writer.flush()
        writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
