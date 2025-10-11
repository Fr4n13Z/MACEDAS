import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from Baselines import set_seed, obtain_logger, FFM_detail_evaluation, text_preprocessing, log_namespace
from Baselines.AttRCNN import AttRCNNModel, AttRCNNDataset


def collate_fn(batch):
    ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    ids = torch.stack(ids)
    labels = torch.tensor(labels)

    return ids, labels


def preprocess_data(args: argparse.Namespace, test_signal: bool = False):
    # load the raw data
    with open(args.raw_data_path, 'r') as f:
        data = json.load(f)
        if args.text_type.lower() == 'segment':
            texts = [[item['Segment'] for item in items if item['Segment'] != ''] for items in data['segments']]
        elif args.text_type.lower() == 'facebook':
            texts = [[text_preprocessing(text).strip() for text in sum(items, []) if text_preprocessing(text) != ''] for items in data['synthesized_texts']]
        elif args.text_type.lower() == 'reddit':
            texts = [[text_preprocessing(text).strip() for text in sum(items, []) if text_preprocessing(text) != ''] for items in data['synthesized_texts']]
        elif args.text_type.lower() == 'essay':
            texts = [[item] for item in data['essay']]
        else:
            raise NotImplementedError("The text type is not supported")
        texts = [item_texts[:args.max_posts] for item_texts in texts]
        labels = data[args.label_type]
    if test_signal is True:
        return texts[:20], labels[:20]
    else:
        return texts, labels


def train_epoch(args: argparse.Namespace, model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_predicted = []
    for feature, labels in tqdm(dataloader, desc="Training"):
        feature, labels = feature.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(feature)
        logits = torch.softmax(logits, dim=-1)[:, 1].view(-1, args.num_labels)

        loss = criterion(logits, labels)
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_labels.extend(labels.detach().cpu().numpy())
        all_predicted.extend(logits.detach().cpu().numpy())
    return total_loss / len(dataloader), all_labels, all_predicted


def evaluate(args: argparse.Namespace, model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for feature, labels in tqdm(dataloader, desc="Evaluating"):
            feature, labels = feature.to(device), labels.to(device)

            logits = model(feature)
            logits = torch.softmax(logits, dim=-1)[:, 1].view(-1, args.num_labels)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 收集所有预测和标签用于F1计算
            all_labels.extend(labels.detach().cpu().numpy())
            all_predicted.extend(logits.detach().cpu().numpy())
        return total_loss / len(dataloader), all_labels, all_predicted


def parse_options(name: str = "AttRCNN"):
    parser = argparse.ArgumentParser(description=f'The parameters for {name} config')
    # dataset object parameters
    parser.add_argument("--raw_data_path", type=str, default="./BenchmarkData/LEDASM/QwQ-32B_facebook.json",
                        help="The path of the cleaned raw data file")
    parser.add_argument('--text_type', type=str, default='facebook', choices=['segment', 'facebook', 'essay', 'reddit'], help='the text types')
    parser.add_argument("--label_type", type=str, default='aligned_labels',
                        choices=['aligned_labels', 'labels'],
                        help="The type of labels used in the dataset")
    parser.add_argument("--process_type", type=str, default="train", choices=['train', 'test', 'apply'],
                        help="The process type")
    parser.add_argument('--max_length', type=int, default=100, help='The maximum length of single post')
    parser.add_argument('--max_posts', type=int, default=40, help="The maximum number of user posts")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size of the DataLoader")
    # BERT model parameters
    parser.add_argument('--bert_path', type=str, default="./PTM/bert-base-uncased",
                        help="The Pretrained BERT model & Tokenizer & Config directory")
    # normal model parameters
    parser.add_argument('--model_name', type=str, default='AttRCNN',
                        help="The name of the personality detection model")
    parser.add_argument('--embedding_dim', type=int, default=768, help="The embedding dimension")
    parser.add_argument('--hidden_dim', type=int, default=20, help="The GRU hidden dimension")
    parser.add_argument('--n_layers', type=int, default=1, help="The GRU layer number")
    parser.add_argument("--gpu_id", type=int, default=0, help="The gpu id")
    # normal processor parameters
    parser.add_argument("--num_epochs", type=int, default=50, help="The epoch of training process")
    parser.add_argument('--num_labels', type=int, default=5, help="The number of labels in the dataset")
    parser.add_argument("--logger_path", type=str, default="./Results/AttRCNN/AttRCNN_Essays_MSE",
                        # QwQ_32B_facebook
                        help="The logging path")
    # other hyper-parameters
    parser.add_argument("--seed", type=int, default=42, help="The root seed for random, pytorch library")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="The linear classifier learning rate")
    parser.add_argument('--save_best_fold_model', type=bool, default=False,
                        help="save the best fold model results")

    parser.add_argument('--kfold_num', type=int, default=5, help="The number of folds for k-fold cross-validation")
    args = parser.parse_args()
    return args


def train_model(args: argparse.Namespace):
    set_seed(args.seed)
    text_list, labels = preprocess_data(args)
    args.logger_path = args.logger_path + f"_{args.text_type}" + f"_{args.label_type}"

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    logger = obtain_logger(logger_path=args.logger_path + ".log", logger_name=__name__)
    log_namespace(logger, args, title="Arguments")
    logger.info('task-process-pid = {}'.format(os.getpid()))
    dataset = AttRCNNDataset(args, text_list, labels)


    kfold = KFold(n_splits=args.kfold_num, shuffle=True, random_state=args.seed)
    fold_acc_results = []
    fold_f1_results = []
    global_best_f1 = 0.
    global_best_fold = 0
    train_result_list = []
    test_result_list = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"\nFold {fold + 1}/{args.kfold_num}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)

        model = AttRCNNModel(args, args.device).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(lr=args.learning_rate, params=model.parameters())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        best_val_f1 = 0
        best_val_acc = 0

        for epoch in range(args.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")

            train_loss, train_labels, train_predicted = train_epoch(args, model, train_loader, criterion, optimizer,
                                                                    args.device)
            train_acc, train_f1, current_train_result_dict = FFM_detail_evaluation(fold, epoch, train_predicted,
                                                                                   train_labels, logger)
            val_loss, test_labels, test_predicted = evaluate(args, model, val_loader, criterion, args.device)
            val_acc, val_f1, current_test_result_dict = FFM_detail_evaluation(fold, epoch, test_predicted, test_labels,
                                                                              logger)
            scheduler.step(val_acc)

            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > global_best_f1 and args.save_best_fold_model:
                global_best_f1 = val_f1
                best_model_state = model.state_dict()
                torch.save(best_model_state, f"{args.logger_path}_best_model.pth")
                global_best_fold = fold

            if val_acc > best_val_acc:
                current_test_result_dict['best_ACC'] = True
                best_val_acc = val_acc
            if val_f1 > best_val_f1:
                current_test_result_dict['best_F1'] = True
                best_val_f1 = val_f1
            train_result_list.append(current_train_result_dict)
            test_result_list.append(current_test_result_dict)

        if global_best_fold == fold and args.save_best_fold_model:
            with open(args.logger_path + ".data", 'wb') as f:
                pickle.dump({'test_data': val_subset}, f)

        fold_acc_results.append(best_val_acc)
        fold_f1_results.append(best_val_f1)
        logger.info(f"Fold {fold + 1} completed. Best Val Acc: {best_val_acc:.4f} | Current Val F1: {best_val_f1:.4f}")

    logger.info(f"\n{args.kfold_num}-fold Cross Validation Results:")
    for fold, item in enumerate(zip(fold_acc_results, fold_f1_results)):
        acc, f1 = item
        logger.info(f"Fold {fold + 1}: Val Acc = {acc:.4f} | Val F1 = {f1:.4f}")
    logger.info(f"Mean Accuracy: {np.mean(fold_acc_results):.4f} ± {np.std(fold_acc_results):.4f}")
    logger.info(f"Mean F1: {np.mean(fold_f1_results):.4f} ± {np.std(fold_f1_results):.4f}")
    train_result_df = pd.DataFrame(train_result_list)
    test_result_df = pd.DataFrame(test_result_list)
    train_result_df.to_csv(args.logger_path + "_train_result.csv", index=False, index_label=False)
    test_result_df.to_csv(args.logger_path + "_test_result.csv", index=False, index_label=False)


def eval_best_model(args: argparse.Namespace):
    args.logger_path = args.logger_path + f"_{args.text_type}" + f"_{args.label_type}"

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    set_seed(args.seed)

    model_dict = torch.load(f"{args.logger_path}_best_model.pth", map_location=device, weights_only=True)
    model = AttRCNNModel(args, device)
    model.load_state_dict(model_dict)
    model = model.to(device)
    criterion = nn.MSELoss()
    with open(args.logger_path + ".data", 'rb') as f:
        data = pickle.load(f)
        val_subset = data['test_data']
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    logger = obtain_logger(logger_path="./temp_test.log", logger_name=__name__)

    model.eval()
    with torch.no_grad():
        val_loss, test_labels, test_predicted = evaluate(args, model, val_loader, criterion, args.device)
        val_acc, val_f1 = FFM_detail_evaluation(0, 0, test_predicted, test_labels, logger)


if __name__ == "__main__":
    args = parse_options()
    if args.process_type == "train":
        train_model(args)
    elif args.process_type == "test":
        eval_best_model(args)
    else:
        print("Invalid process type!")
