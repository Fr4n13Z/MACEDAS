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
from Baselines.DDGCN import DDGCN, DDGCNDataset, get_optimizer

label_dict = {'D': 0, 'I': 1, 'S': 2, 'C': 3}


# 自定义collate_fn处理变长序列
def collate_fn(batch):
    ids = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    ids = torch.stack(ids)
    masks = torch.stack(masks)
    labels = torch.tensor(labels)

    return ids, masks, labels


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


# 训练和评估函数
def train_epoch(args: argparse.Namespace, model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_predicted = []
    for post_ids, post_mask, labels in tqdm(dataloader, desc="Training"):
        post_ids, post_mask, labels = post_ids.to(device), post_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(post_ids, post_mask)
        logits = torch.stack(outputs[:-2], dim=1).view(-1, 2)
        logits = torch.softmax(logits, dim=-1)[:, 1].view(-1, args.num_labels)  # 获取所有logits
        l0_attr, rs = outputs[-2], outputs[-1]
        total_l0loss = 0.
        batch_samples = labels.size(0)
        for i in range(args.num_labels):
            total_l0loss += l0_attr[i][0]
        total_l0loss /= (args.num_labels * batch_samples)
        loss = criterion(logits, labels)

        if args.l0:
            optimizer.update(total_l0loss, loss, model)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
        if not args.l0:
            optimizer.step()
        total_loss += loss.item()
        # 收集所有预测和标签用于F1计算
        all_labels.extend(labels.detach().cpu().numpy())
        all_predicted.extend(logits.detach().cpu().numpy())
    return total_loss / len(dataloader), all_labels, all_predicted


def evaluate(args: argparse.Namespace, model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for post_ids, post_mask, labels in tqdm(dataloader, desc="Evaluating"):
            post_ids, post_mask, labels = post_ids.to(device), post_mask.to(device), labels.to(device)
            outputs = model(post_ids, post_mask)
            logits = torch.stack(outputs[:-2], dim=1).view(-1, 2)  # 获取所有logits
            logits = torch.softmax(logits, dim=-1)[:, 1].view(-1, args.num_labels)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            # 收集所有预测和标签用于F1计算
            all_labels.extend(labels.detach().cpu().numpy())
            all_predicted.extend(logits.detach().cpu().view(-1, args.num_labels).numpy())
    return total_loss / len(dataloader), all_labels, all_predicted


def parse_options(name: str = "D-DGCN"):
    parser = argparse.ArgumentParser(description=f'The parameters for {name} config')
    # dataset object parameters
    parser.add_argument("--raw_data_path", type=str,
                        default="./BenchmarkData/LEDASM/QwQ-32B_facebook.json",
                        help="The path of the cleaned raw data file")

    parser.add_argument("--label_type", type=str, default='aligned_labels',
                        choices=['aligned_labels', 'labels'],
                        help="The type of labels used in the dataset")
    parser.add_argument("--process_type", type=str, default="train", choices=['train', 'test', 'apply'],
                        help="The process type")
    parser.add_argument('--text_type', type=str, default='facebook', choices=['segment', 'facebook', 'essay', 'reddit'], help='the text types')
    # max_posts for segment: 20 max_posts for facebook posts: 40
    # max length for segment: 140, max length for facebook posts: 100
    parser.add_argument('--max_length', type=int, default=100, help='The maximum length of single post')
    parser.add_argument('--max_posts', type=int, default=40, help="The maximum number of user posts")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size of the DataLoader")
    parser.add_argument('--num_labels', type=int, default=5, help="The number of labels in the dataset")

    # BERT model parameters
    parser.add_argument('--ptm_path', type=str, default="./PTM/bert-base-uncased",
                        help="The Pretrained BERT model & Tokenizer & Config directory")

    # normal model parameters
    parser.add_argument('--hidden_size', type=int, default=768, help="The embedding dimension")
    parser.add_argument('--gcn_hidden_size', type=int, default=768, help="The hidden dimension of the DGCN")
    parser.add_argument('--threshold', type=float, default=0.5, help="The AdjMatrix Threshold")
    parser.add_argument('--gcn_dropout', type=float, default=0.2, help='Dropout rate for GCN.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for embedding.')
    parser.add_argument('--final_hidden_size', type=int, default=128, help="Hidden size of Attention Cells")
    parser.add_argument('--mlp_num', type=int, default=2, help='Number of MLP Layer in the last of model.')
    parser.add_argument('--embedding_dim', type=int, default=768, help="Dynamic_DeeperGCN Embedding Dimension")
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes of each personality dimension.')
    parser.add_argument("--gnn_num_layers", type=int, default=2, help="Number of GCN Layers")

    # normal processor parameters
    parser.add_argument("--gpu_id", type=int, default=0, help="The gpu id")
    parser.add_argument("--num_epochs", type=int, default=10, help="The epoch of training process")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='the number of accumulation steps of gradient')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maxinum of gradient')
    parser.add_argument('--l0', type=bool, default=True, help='whether to use L0 loss function')
    parser.add_argument('--single_hop', type=bool, default=False, help='single_hop variant for ablation experiments')

    # other hyper-parameters
    parser.add_argument('--no_special_node', type=bool, default=False, help="The ablation experiments or not in D-DGCN")
    parser.add_argument("--seed", type=int, default=42, help="The root seed for random, pytorch library")
    parser.add_argument("--logger_path", type=str, default="./Results/DDGCN/DDGCN_Essays_MSE", help="The logging path")

    # learning rates
    parser.add_argument("--other_lr", default=1e-4, type=float, help="The initial learning rate for other components.")
    parser.add_argument('--alpha_lr', default=1e-2, type=float, help="alpha_optimizer_lr in LagrangianOptimization")
    parser.add_argument("--gm_lr", default=1e-5, type=float, help="L2C learning rate")
    parser.add_argument('--pretrained_model_lr', type=float, default=2e-5, help="The pretrained model learning rate")
    parser.add_argument('--max_alpha', default=100, type=int, help="max_alpha in LagrangianOptimization")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--save_best_fold_model', type=bool, default=False, help="save the best fold model results")

    parser.add_argument('--kfold_num', type=int, default=5, help="The number of folds for k-fold cross-validation")
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
    dataset = DDGCNDataset(args, text_list, labels)

    # 10折交叉验证
    kfold = KFold(n_splits=args.kfold_num, shuffle=True, random_state=args.seed)
    fold_acc_results = []
    fold_f1_results = []
    global_best_fold = 0
    global_best_f1 = 0.
    train_result_list = []
    test_result_list = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"\nFold {fold + 1}/ {args.kfold_num}")

        # 初始化模型
        model = DDGCN(args, device).to(device)

        # 创建数据子集和数据加载器
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)
        criterion = nn.MSELoss()
        optimizer = get_optimizer(args, model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer.original_optimizer, 'max', patience=2)

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
    model = DDGCN(args, device)
    model.load_state_dict(model_dict)
    model = model.to(device)
    criterion = nn.MSELoss()
    with open(args.logger_path + ".data", 'rb') as f:
        data = pickle.load(f)
        val_subset = data['test_data']
        print(f"测试集规模：{len(val_subset)}")

    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    logger = obtain_logger(logger_path="./temp_test.log", logger_name=__name__)

    # 评估模型
    model.eval()
    with torch.no_grad():
        val_loss, test_labels, test_predicted = evaluate(args, model, val_loader, criterion, args.device)
        FFM_detail_evaluation(0, 0, test_predicted, test_labels, logger)


if __name__ == '__main__':
    # 预设参数
    args = parse_options()
    if args.process_type == "train":
        train_model(args)
    elif args.process_type == "test":
        eval_best_model(args)
    else:
        print("Invalid process type!")
