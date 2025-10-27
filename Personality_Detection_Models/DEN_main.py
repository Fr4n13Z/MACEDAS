import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from Baselines import set_seed, obtain_logger, FFM_detail_evaluation, text_preprocessing, log_namespace
from Baselines.DEN import DENDataset, DENModel


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
        texts = [item_texts[:args.max_posts] if len(item_texts) > args.max_posts else item_texts + [""] * (
                    args.max_posts - len(item_texts)) for item_texts in texts]
        labels = data[args.label_type]
    if test_signal is True:
        return texts[:20], labels[:20]
    else:
        return texts, labels


# 训练一个epoch
def train_epoch(args, model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_predicted = []
    with torch.enable_grad():
        for batch in tqdm(dataloader, desc="Training"):
            post_ids = batch.post_ids.to(device)
            post_mask = batch.post_mask.to(device)
            word_nodes = batch.word_nodes.to(device)
            word_mask = batch.word_mask.to(device)
            word2word_index = batch.word2word_index.to(device)
            post2word_index = batch.post2word_index.to(device)
            labels = batch.label.to(device).view(-1, args.num_labels)

            logits = model(post_ids, post_mask, word_nodes, word_mask, word2word_index, post2word_index)
            logits = torch.softmax(logits, dim=-1)[:, 1].view(-1, args.num_labels)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            all_labels.extend(labels.detach().cpu().numpy())
            all_predicted.extend(logits.detach().cpu().numpy())
    return total_loss / len(dataloader), all_labels, all_predicted


def evaluate(args, model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            post_ids = batch.post_ids.to(device)
            post_mask = batch.post_mask.to(device)
            word_nodes = batch.word_nodes.to(device)
            word_mask = batch.word_mask.to(device)
            word2word_index = batch.word2word_index.to(device)
            post2word_index = batch.post2word_index.to(device)
            labels = batch.label.to(device).view(-1, args.num_labels)

            logits = model(post_ids, post_mask, word_nodes, word_mask, word2word_index, post2word_index)
            logits = torch.softmax(logits, dim=-1)[:, 1].view(-1, args.num_labels)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_labels.extend(labels.detach().cpu().numpy())
            all_predicted.extend(logits.detach().cpu().numpy())
    return total_loss / len(dataloader), all_labels, all_predicted


def parse_options(name: str = "DEN"):

    parser = argparse.ArgumentParser(description=f'The parameters for {name} config')

    parser.add_argument("--process_type", type=str, default="train", choices=['train', 'test', 'apply'],
                        help="The process type")
    parser.add_argument("--raw_data_path", type=str, default="./BenchmarkData/LEDASM/QwQ-32B_facebook.json",
                        help="The path of the cleaned raw data file")
    parser.add_argument('--text_type', type=str, default='facebook', choices=['segment', 'facebook', 'essay', 'reddit'], help='the text types')
    parser.add_argument("--label_type", type=str, default='aligned_labels',
                        choices=['aligned_labels', 'labels', 'local_shift', 'global_shift'],
                        help="The type of labels used in the dataset")
    parser.add_argument("--max_word_num", type=int, default=480, help="The maximum unique entity number of user")
    parser.add_argument('--max_posts', type=int, default=40, help="The maximum number of user posts")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size of the DataLoader")
    parser.add_argument('--num_labels', type=int, default=5, help="The number of labels in the dataset")

    parser.add_argument('--ptm_path', type=str, default="./PTM/bert-base-uncased",
                        help="The Pretrained model & Tokenizer & Config directory")
    parser.add_argument("--glove_path", type=str, default="./PTM/GloVE/glove.42B.300d.word2vec")
    parser.add_argument('--liwc_path', type=str, default="./PTM/LIWC2015_English.dic")
    # normal model parameters
    parser.add_argument('--glove_embed_dim', type=int, default=300, help="The embedding dimension of GloVe embeddings")
    parser.add_argument('--hidden_dim', type=int, default=768, help="The embedding dimension")
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0, help="The gpu id")
    parser.add_argument('--alpha', type=float, default=0.5, help="The alpha for the final user representation")
    parser.add_argument('--freeze_bert', type=bool, default=False, help="freeze the ptm model parameters")
    # normal processor parameters
    parser.add_argument("--num_epochs", type=int, default=10, help="The epoch of training process")
    parser.add_argument("--logger_path", type=str, default="./Results/DEN/DEN_Essays_MSE", help="The logging path")
    # other hyper-parameters
    parser.add_argument("--seed", type=int, default=42, help="The root seed for random, pytorch library")
    parser.add_argument('--linear_lr', type=float, default=1e-3, help="The linear classifier learning rate")
    parser.add_argument('--pretrained_model_lr', type=float, default=2e-5,
                        help="The pretrained model learning rate")
    parser.add_argument('--save_best_fold_model', type=bool, default=False,
                        help="save the best fold model Results")
    parser.add_argument('--kfold_num', type=int, default=10, help="The number of folds for k-fold cross-validation")
    args = parser.parse_args()
    return args


def train_model(args: argparse.Namespace):
    set_seed(args.seed)
    text_list, labels = preprocess_data(args)
    args.logger_path = args.logger_path + f"_{args.text_type}" + f"_{args.label_type}"
    # GPU设置
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    # 日志文件保存
    logger = obtain_logger(logger_path=args.logger_path + '.log', logger_name=__name__)
    log_namespace(logger, args, title="Arguments")
    logger.info('task-process-pid = {}'.format(os.getpid()))
    dataset = DENDataset(args, text_list, labels)

    # 10折交叉验证
    kfold = KFold(n_splits=args.kfold_num, shuffle=True, random_state=args.seed)
    fold_acc_results = []
    fold_f1_results = []
    global_best_fold = 0
    global_best_f1 = 0.
    train_result_list = []
    test_result_list = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"\nFold {fold + 1}/{args.kfold_num}")
        # 初始化模型
        model = DENModel(args).to(device)
        # 创建数据子集和数据加载器
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False)
        criterion = nn.MSELoss()
        ptm_id = list(map(id, model.post_encoder.parameters()))
        other_params = filter(lambda p: id(p) not in ptm_id, model.parameters())
        optimizer_grouped_parameters = [
            {'params': other_params, 'lr': args.linear_lr}]

        if args.freeze_bert is False:
            optimizer_grouped_parameters += [{'params': model.post_encoder.parameters(), 'lr': args.pretrained_model_lr}]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

        best_val_acc = 0
        best_val_f1 = 0
        # 训练循环
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
            logger.info(
                f"Avg Train Loss: {train_loss:.4f} | Avg Train Acc: {train_acc:.4f} | Avg Train F1: {train_f1:.4f}")
            logger.info(f"Avg Val Loss: {val_loss:.4f} | Avg Val Acc: {val_acc:.4f} | Avg Val F1: {val_f1:.4f}")
            if val_f1 > global_best_f1 and args.save_best_fold_model:
                global_best_f1 = val_f1
                best_model_state = model.state_dict()
                torch.save(best_model_state, f"{args.logger_path}_best_model.pth")
                global_best_fold = fold

            # 保存最佳结果
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
        logger.info(
            f"Fold {fold + 1} completed. Best Val Acc: {best_val_acc:.4f} | Current Val F1: {best_val_f1:.4f}")

    # 输出最终结果
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
    # GPU设置
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    # 加载最佳模型
    model_dict = torch.load(f"{args.logger_path}_best_model.pth", map_location=device, weights_only=True)
    model = DENModel(args)
    model.load_state_dict(model_dict)
    model = model.to(device)
    criterion = nn.MSELoss()
    with open(args.logger_path + ".data", 'rb') as f:
        data = pickle.load(f)
        val_subset = data['test_data']
        print(f"测试集规模：{len(val_subset)}")
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False)
    logger = obtain_logger(logger_path="./temp_test.log", logger_name=__name__)
    # 评估模型
    model.eval()
    with torch.no_grad():
        val_loss, test_labels, test_predicted = evaluate(args, model, val_loader, criterion, args.device)
        val_acc, val_f1, result_dict = FFM_detail_evaluation(0, 0, test_predicted, test_labels, logger)


if __name__ == '__main__':
    # 预设参数
    args = parse_options()
    if args.process_type == "train":
        train_model(args)
    elif args.process_type == "test":
        eval_best_model(args)
    else:
        print("Invalid process type!")
