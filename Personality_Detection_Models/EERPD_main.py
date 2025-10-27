import argparse
import json
import os
import pickle

import torch
from sklearn.model_selection import KFold

from Baselines import text_preprocessing, obtain_logger, log_namespace
from Baselines.EERPD import local_eer_process, construct_preference_library, few_shot_cot_process, prediction
from Baselines.EERPD.dataset import EERPDDataset


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


def parse_options(name: str = "EERPD"):
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument('--api_key', type=str, default='Empty', help="The API key for LLM")
    parser.add_argument('--seed', type=int, default=42, help='the random seed')
    parser.add_argument('--base_url', type=str, default='http://192.168.2.150:54321', help='the base url for LLM')
    parser.add_argument('--raw_data_path', type=str, default='./BenchmarkData/LEDASM/QwQ-32B_facebook.json', help='the path of raw data')
    parser.add_argument('--model', type=str, default="Qwen3-32B", help='The LLM name')
    parser.add_argument('--text_type', type=str, default='facebook', choices=['segment', 'facebook', 'essay', 'reddit'], help='the text types')
    parser.add_argument("--label_type", type=str, default='aligned_labels',
                        choices=['aligned_labels', 'labels', 'local_shift', 'global_shift'],
                        help="The type of labels used in the dataset")
    parser.add_argument("--gpu_id", type=int, default=0, help="The gpu id")
    parser.add_argument('--max_posts', type=int, default=40)
    parser.add_argument('--ptm_path', type=str, default="./PTM/RoBERTa-base-uncased",
                        help="The Pretrained BERT model & Tokenizer & Config directory")
    parser.add_argument('--num_workers', type=int, default=8, help='the number of LLM calling worker')
    parser.add_argument('--logger_path', type=str, default='./Results/EERPD/EERPD_finegrained')

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--kfold_num', type=int, default=10, help="The number of folds for k-fold cross-validation")

    parser.add_argument("--without_kfold", type=bool, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_options()
    args.logger_path = args.logger_path + f"_{args.text_type}" + f"_{args.label_type}"
    # preprocess the data
    text_list, labels = preprocess_data(args)
    # 日志文件保存
    logger = obtain_logger(logger_path=args.logger_path + '.log', logger_name=__name__)
    logger.info('task-process-pid = {}'.format(os.getpid()))
    log_namespace(logger, args, title="Arguments")
    dataset = EERPDDataset(args, text_list, labels)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    args.device = device


    # 10折交叉验证
    kfold = KFold(n_splits=args.kfold_num, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"\nFold {fold + 1}/ {args.kfold_num}")
        train_texts = [dataset[idx][0] for idx in train_idx]
        train_labels = [dataset[idx][1] for idx in train_idx]
        val_texts = [dataset[idx][0] for idx in val_idx]
        val_labels = [dataset[idx][1] for idx in val_idx]
        # construct the preference library
        """
        with open(args.logger_path + f"_{args.text_type}_eer_knowledge.json", 'r') as f:
            eer_knowledge = json.load(f)
        """
        eer_knowledge = local_eer_process(args=args,
                                               text_list=train_texts,
                                               output_path=args.logger_path + f"_{args.text_type}_eer_knowledge.json")
        """with open(args.logger_path + f"_{args.text_type}_eer_test_samples.json", 'r') as f:
            eer_test_sample_knowledge = json.load(f)"""
        eer_test_sample_knowledge = local_eer_process(args=args,
                                               text_list=val_texts,
                                               output_path=args.logger_path + f"_{args.text_type}_eer_test_samples.json")
        """
        with open(args.logger_path + f"_{args.text_type}_ref_library.pkl", 'rb') as f:
            reference_library = pickle.load(f)
        with open(args.logger_path + f"_{args.text_type}_test_sample_ref_library.pkl", 'rb') as f:
            test_sample_reference_library = pickle.load(f)
        """
        reference_library = construct_preference_library(args=args,
                                                         text_list=train_texts,
                                                         labels=train_labels,
                                                         eer_results=eer_knowledge,
                                                         save_path=args.logger_path + f"_{args.text_type}_ref_library.pkl")
        test_sample_reference_library = construct_preference_library(args=args,
                                                                     text_list=val_texts,
                                                                     labels=val_labels,
                                                                     eer_results=eer_test_sample_knowledge,
                                                                     save_path=args.logger_path + f"_{args.text_type}_test_sample_ref_library.pkl")
        """
        with open(args.logger_path + f"_{args.text_type}_test_sample_cot_samples.json", 'r') as f:
            few_shot_cot_samples = json.load(f)"""
        few_shot_cot_samples = few_shot_cot_process(args=args,
                                                    reference_library=reference_library,
                                                    test_sample_library=test_sample_reference_library,
                                                    output_path=args.logger_path + f"_{args.text_type}_test_sample_cot_samples.json")
        prediction_results = prediction(args=args,
                                        test_sample_library=test_sample_reference_library,
                                        few_shot_cot_samples=few_shot_cot_samples,
                                        output_path=args.logger_path + f"_{args.text_type}_test_sample_prediction.json")
        if args.without_kfold:
            break


