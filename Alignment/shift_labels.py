import argparse
import copy
import json
import pickle
import re
import statistics
from typing import Optional
import pandas as pd

from .capture_facets import facet_correction


IPIP_NEO_facet_correlation_dict = {
    'anxiety': 0.92,
    'anger': 0.92,
    'depression': 0.94,
    'self-consciousness': 0.95,
    'immoderation': 0.99,
    'vulnerability': 0.97,
    'friendliness': 0.91,
    'gregariousness': 0.98,
    'assertiveness': 0.99,
    'activity level': 0.99,
    'excitement-seeking': 0.95,
    'cheerfulness': 0.95,
    'imagination': 0.91,
    'artistic interests': 0.95,
    'emotionality': 0.91,
    'adventurousness': 0.99,
    'intellect': 0.96,
    'liberalism': 0.87,
    'trust': 0.94,
    'morality': 0.88,
    'altruism': 0.91,
    'cooperation': 0.99,
    'modesty': 0.95,
    'sympathy': 0.92,
    'self-efficacy': 0.91,
    'orderliness': 0.98,
    'dutifulness': 0.86,
    'achievement-striving': 0.94,
    'self-discipline': 0.93,
    'cautiousness': 0.95
}


big5_type_dict = {'E': 0, 'N': 1, 'A': 2, 'C': 3, 'O': 4}


res2rate_dict = {'Agree': 1., 'Disagree': 0., 'Neutral': 0.5, 'Partially Agree': 0.75, 'Partially Disagree': 0.25}


def align_labels_via_global_and_local_views(args: argparse.Namespace,
                                            local_data: dict | Optional[str],
                                            global_data: dict | Optional[str],
                                            item_path: str,
                                            output_path: Optional[str]=None):
    # load item dataframe
    item_df = pd.read_csv(item_path)
    item_df['trait'] = item_df['Key&Num.'].apply(lambda x: x[0])
    item_df['Key'] = item_df['Key'].apply(lambda x: x.lower())
    item_list = list(set(item_df['Key'].tolist()))

    # load scores from local data
    if isinstance(local_data, str):
        if local_data.endswith(".json"):
            with open(local_data, 'r') as f:
                local_data = json.load(f)
        else:
            with open(local_data, 'rb') as f:
                local_data = pickle.load(f)
    local_scores = local_data['local_scoring']

    # load scores from global data
    if isinstance(global_data, str):
        if global_data.endswith(".json"):
            with open(global_data, 'r') as f:
                global_data = json.load(f)
        else:
            with open(global_data, 'rb') as f:
                global_data = pickle.load(f)
    global_scores = global_data['global_scoring']

    labels = local_data['labels']
    assert len(global_scores) == len(local_scores) ==len(labels)

    # facet-trait convertion mapping
    facet2trait_dict = {}
    for facet, trait in zip(item_df['Key'].tolist(), item_df['trait'].tolist()):
        if facet not in facet2trait_dict:
            facet2trait_dict[facet] = trait
    item_stats = item_df.groupby('Key')

    # facet-positive statement mapping
    facet2pos_dict = {}
    for facet, group in item_stats:
        pos_statements = group[group['Sign'] == 1]
        facet2pos_dict[facet] = [1 for _ in range(pos_statements.shape[0])]
        facet2pos_dict[facet] += [-1 for _ in range(group[group['Sign'] == -1].shape[0])]

    # facet word correction check
    local_score_facets = list({key.lower() for segments in local_scores for segment in segments for key in segment})
    global_score_facets = list({key.lower() for segments in global_scores for segment in segments for key in segment})
    score_keys = list(set(local_score_facets + global_score_facets + item_list))
    if len(score_keys) != len(item_list):
        facet2correct_facet_dict = facet_correction(args.api_key, args.model, args.llm_url, score_keys)
    else:
        facet2correct_facet_dict = {item: item for item in item_list}

    # compute local assessment scores
    local_results = [[[], [], [], [], []] for _ in range(len(local_scores))]
    for idx, seg_scores in enumerate(local_scores):
        for jdx, seg_score in enumerate(seg_scores):
            for facet in seg_score.keys():
                correct_facet = facet2correct_facet_dict[facet.lower()]
                facet_cor_para = IPIP_NEO_facet_correlation_dict[correct_facet]
                facet_trait_dim = big5_type_dict[facet2trait_dict[correct_facet]]
                score_result = 0.
                if len(list(set(seg_score[facet]))) == 1 and seg_score[facet][0] == 'Neutral':
                    continue
                for score, attr in zip(seg_score[facet], facet2pos_dict[correct_facet]):
                    if attr == 1:
                        score_result = score_result + res2rate_dict[score] * facet_cor_para
                    else:
                        score_result = score_result + (1 - res2rate_dict[score]) * facet_cor_para
                local_results[idx][facet_trait_dim].append(score_result/len(seg_score[facet]))
    local_assess_scores = [[statistics.mean(item[i]) if item[i] != [] else labels[idx][i] for i in range(5)] for idx, item
                        in enumerate(local_results)]

    # compute global assessment scores
    global_results = [[[], [], [], [], []] for _ in range(len(local_scores))]
    for idx, text_score in enumerate(global_scores):
        if isinstance(text_score, list):
            text_score = text_score[0]
        for facet in text_score.keys():
            correct_facet = facet2correct_facet_dict[facet.lower()]
            facet_cor_para = IPIP_NEO_facet_correlation_dict[correct_facet]
            facet_trait_dim = big5_type_dict[facet2trait_dict[correct_facet]]
            score_result = 0.
            if len(list(set(text_score[facet]))) == 1 and text_score[facet][0] == 'Neutral':
                continue
            for score, attr in zip(text_score[facet], facet2pos_dict[correct_facet]):
                if attr == 1:
                    score_result = score_result + res2rate_dict[score] * facet_cor_para
                else:
                    score_result = score_result + (1 - res2rate_dict[score]) * facet_cor_para
            global_results[idx][facet_trait_dim].append(score_result / len(text_score[facet]))
    global_assess_scores = [[statistics.mean(item[i]) if item[i] != [] else labels[idx][i] for i in range(5)] for idx, item
                         in enumerate(global_results)]

    # label shift to align the original texts in both local and global views
    final_aligned_labels = []
    for label, global_res, local_res in zip(labels, global_assess_scores, local_assess_scores):
        aligned_label = [0.25 * float(global_res[i]) + 0.25 * float(local_res[i]) + 0.5 * float(label[i]) for i in
                         range(5)]
        final_aligned_labels.append(aligned_label)

    # result construction
    final_data = {'author_id': local_data['author_id'] if 'author_id' in local_data.keys() else [],
                  'essay': local_data['text'],
                  'segments': [item[-1] for item in local_data['segments']],
                  'labels': local_data['labels'],
                  'aligned_labels': final_aligned_labels,
                  'global_shift': global_assess_scores,
                  'local_shift': local_assess_scores}

    if output_path is not None:
        if output_path.endswith(".json"):
            with open(output_path, 'w') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=4)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(final_data, f)
    return final_data








