import argparse
import gc
import json
import pickle
import numpy as np
import requests
import torch
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from .prompt import EER_prompt, CoT_prompt, Prediction_prompt

label_dim_dict = {0: 'E', 1: 'N', 2: 'A', 3: 'C', 4: 'O'}


def label_value_to_string(label_value: int | float):
    if label_value > 0.75:
        return 'High'
    elif label_value > 0.5:
        return 'Partially High'
    elif label_value > 0.25:
        return 'Partially Low'
    else:
        return 'Low'


def label_process(label: List[int]):
    label_dict = {label_dim_dict[dim]: label_value_to_string(label_value) for dim, label_value in enumerate(label)}
    return json.dumps(label_dict)


def extract_code_from_python(content):
    pattern = r'```python\s*(.*?)```'
    result = re.search(pattern, content.strip(), re.DOTALL)
    if result:
        extracted_code = result.group(1).strip()
    else:
        extracted_code = content.strip()
        print("[WARNING] No python code found.")
    return extracted_code


def extract_code_from_json(content):
    pattern = r'```json\s*(.*?)```'
    result = re.search(pattern, content.strip(), re.DOTALL)
    if result:
        extracted_code = result.group(1).strip()
    else:
        extracted_code = content.strip()
        print("[WARNING] No python code found.")
    return extracted_code


def classify_emotion_items(text: str):
    results = []
    eer_list = text.split("\n")
    for item in eer_list:
        if 'regulation' in item.strip().lower():
            results.append('Emotion Regulation')
        else:
            results.append('Emotion')
    return results


def local_llm_request(model: str,
                      messages: List[Dict[str, str]],
                      url: str,
                      verify: Optional[bool] = False,
                      timeout: Optional[int] = None,
                      stream: Optional[bool] = False):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer EMPTY'
    }

    # 改进URL拼接逻辑
    if not url.endswith("/v1/chat/completions"):
        url = url.rstrip('/') + "/v1/chat/completions"

    payload_dict = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": 0.7,
    }

    try:
        payload = json.dumps(payload_dict, ensure_ascii=False)
        response = requests.post(
            url=url,
            data=payload.encode('utf-8'),
            headers=headers,
            verify=verify,
            timeout=timeout or 600  # 设置默认超时
        )
        response.raise_for_status()  # 检查HTTP错误

        response_dict = response.json()

    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        print(error_msg)
        return error_msg, error_msg, 0
    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {str(e)}"
        print(error_msg)
        return error_msg, error_msg, 0
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return error_msg, error_msg, 0

    # 更健壮的响应检查
    if 'choices' not in response_dict or not response_dict['choices']:
        error_msg = f"Invalid response format: {response_dict}"
        print(error_msg)
        return error_msg, error_msg, 0

    choice = response_dict['choices'][0]
    if 'message' not in choice:
        error_msg = f"No message in response: {response_dict}"
        print(error_msg)
        return error_msg, error_msg, 0

    message = choice['message']

    # 安全地获取字段值
    reasoning_content = message.get('reasoning_content', '')
    content = message.get('content', '')
    total_tokens = response_dict.get('usage', {}).get('total_tokens', 0)

    return reasoning_content, content, total_tokens


def thread_eer_llm_call(idx: int, model: str, base_url: str, messages: List[Dict[str, str]], length: int):
    _, response, _ = local_llm_request(model, messages, base_url, verify=False, timeout=None, stream=False)
    results = classify_emotion_items(response.strip())
    flag = False
    if len(results) != length:
        flag = True
    return idx, results, flag

def thread_cot_llm_call(idx: int, model: str, url: str, messages: List[Dict[str, str]]):
    cot_content, response, tokens = local_llm_request(model, messages, url)
    return idx, response, tokens


def thread_prediction_llm_call(idx: int, model: str, url: str, messages: List[Dict[str, str]]):
    cot_content, response, tokens = local_llm_request(model, messages, url)
    flag = True
    try:
        if '```json' in response.lower():
            response = extract_code_from_json(response)
        response = eval(response.strip())
        flag = False
    except:
        response = response.replace("```", "")
    return idx, response, flag


def local_eer_process(args: argparse.Namespace, text_list: List[List[str]], output_path: str = None):
    error_list = []
    responses_list = [None for _ in range(len(text_list))]
    messages_list = []
    for user_texts in text_list:
        messages_list.append(
            [{'role': 'user', 'content': EER_prompt.format(posts="|||".join(user_texts), post_num=len(user_texts))}])
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(thread_eer_llm_call, idx, args.model, args.base_url, messages, len(text_list[idx]))
                   for
                   idx, messages in enumerate(messages_list)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="EER Process"):
            idx, response, flag = future.result()
            if flag is False:
                responses_list[idx] = response
            else:
                error_list.append([idx, messages_list[idx], len(text_list[idx])])
    while len(error_list) != 0:
        new_error_list = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(thread_eer_llm_call, item[0], args.model, args.base_url, item[1], item[2]) for
                       item in
                       error_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="EER Process Fixing"):
                idx, response, flag = future.result()
                if flag is False:
                    responses_list[idx] = response
                else:
                    new_error_list.append([idx, messages_list[idx], len(text_list[idx])])
        error_list = new_error_list
    if output_path is not None:
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(responses_list, f)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(responses_list, f)
    return responses_list


def construct_preference_library(args: argparse.Namespace, text_list: List[List[str]], labels: List[List[int]],
                                 eer_results: List[List[str]], save_path: Optional[str] = None):
    embed_model = SentenceTransformer(args.ptm_path, device=args.device, local_files_only=True)
    reference_library = {'vector': [], 'texts': [], 'E_texts': [], 'ER_texts': [], 'labels': labels}
    for i_err_results, user_texts, label in tqdm(zip(eer_results, text_list, labels)):
        i_e_text_list = []
        i_er_text_list = []
        if len(i_err_results) != len(user_texts):
            post_num = min(len(i_err_results), len(user_texts))
            i_err_results = i_err_results[:post_num]
            user_texts = user_texts[:post_num]
        for ij_err_result, j_post in zip(i_err_results, user_texts):
            if 'emotion regulation' in ij_err_result.lower():
                i_er_text_list.append(j_post)
            else:
                i_e_text_list.append(j_post)
        i_e_vector = embed_model.encode("\n".join(i_e_text_list), convert_to_numpy=True)
        i_er_vector = embed_model.encode("\n".join(i_er_text_list), convert_to_numpy=True)
        i_vector = args.alpha * i_e_vector + (1 - args.alpha) * i_er_vector
        reference_library['vector'].append(i_vector)
        reference_library['texts'].append(user_texts)
        reference_library['E_texts'].append(i_e_text_list)
        reference_library['ER_texts'].append(i_er_text_list)
    if save_path is not None:
        if save_path.endswith('.json'):
            with open(save_path, 'w') as f:
                json.dump(reference_library, f)
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(reference_library, f)
    del embed_model
    torch.cuda.empty_cache()
    gc.collect()
    return reference_library


def retrieve_few_shot_example(reference_library: Dict, test_sample_library: Dict):
    test_sample_vectors = np.array(test_sample_library['vector'])
    reference_vectors = np.array(reference_library['vector'])
    similarities = cosine_similarity(test_sample_vectors, reference_vectors)
    cot_examples_idx = [np.argsort(similarities[i])[-2:].tolist() for i in range(similarities.shape[0])]
    return cot_examples_idx


def few_shot_cot_process(args: argparse.Namespace, reference_library: Dict, test_sample_library: Dict,
                         output_path: str = None):
    few_shot_labels = []
    few_shot_texts = []
    cot_samples_idx = retrieve_few_shot_example(reference_library, test_sample_library)
    task_list = []
    task_id_list = []
    count = 0

    for i_cot_sample_idx, user_texts in tqdm(zip(cot_samples_idx, test_sample_library['texts'])):
        e1_texts = "|||".join(reference_library['texts'][i_cot_sample_idx[0]])
        e1_label = reference_library['labels'][i_cot_sample_idx[0]]
        e1_label = label_process(e1_label)
        e2_texts = "|||".join(reference_library['texts'][i_cot_sample_idx[1]])
        e2_label = reference_library['labels'][i_cot_sample_idx[1]]
        e2_label = label_process(e2_label)
        e1_cot_prompt = CoT_prompt.replace("{personality_type}", e1_label).replace("{posts}", e1_texts)
        e2_cot_prompt = CoT_prompt.replace("{personality_type}", e2_label).replace("{posts}", e2_texts)
        task_list += [[{'role': 'user', 'content': e1_cot_prompt}], [{'role': 'user', 'content': e2_cot_prompt}]]
        task_id_list += [[count, 0], [count, 1]]
        few_shot_texts.append([e1_texts, e2_texts])
        few_shot_labels.append([e1_label, e2_label])
        count += 1

    responses_list = [[{}, {}] for _ in range(len(few_shot_texts))]

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(thread_cot_llm_call, idx, args.model, args.base_url, messages) for idx, messages in
                   enumerate(task_list)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Few-Shot CoT Process"):
            idx, response, tokens = future.result()
            responses_list[task_id_list[idx][0]][task_id_list[idx][1]] = {
                'original_texts': few_shot_texts[task_id_list[idx][0]][task_id_list[idx][1]],
                'label': few_shot_labels[task_id_list[idx][0]][task_id_list[idx][1]],
                'cot_content': response.strip(),
                'tokens': tokens}
    if output_path is not None:
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(responses_list, f)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(responses_list, f)
    return responses_list


def prediction(args: argparse.Namespace, test_sample_library: Dict, few_shot_cot_samples: List,
               output_path: str = None):
    task_list = []
    results = [None for _ in range(len(test_sample_library['labels']))]
    error_list = []
    error_try_times = 10
    for user_texts, few_shot_sample in tqdm(zip(test_sample_library['texts'], few_shot_cot_samples)):
        template = Prediction_prompt.replace("{e1_posts}", few_shot_sample[0]['original_texts']).replace(
            "{e1_personality_type}", few_shot_sample[0]['label']).replace(
            "{e1_cot_process}", few_shot_sample[0]['cot_content'].replace("Process:", "")).replace(
            "{e2_posts}", few_shot_sample[1]['original_texts']).replace(
            "{e2_personality_type}", few_shot_sample[1]['label']).replace(
            "{e2_cot_process}", few_shot_sample[1]['cot_content'].replace("Process:", "")).replace(
            "{posts}", "|||".join(user_texts))
        task_list.append([{'role': 'user', 'content': template}])
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(thread_prediction_llm_call, idx, args.model, args.base_url, messages) for idx, messages in
                   enumerate(task_list)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Personality Prediction Process"):
            idx, response, flag = future.result()
            results[idx] = response
            if flag is False:
                results[idx] = response
            else:
                error_list.append([idx, task_list[idx]])
    while len(error_list) > 5:
        new_error_list = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(thread_prediction_llm_call, item[0], args.model, args.base_url, item[1]) for item in
                       error_list]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Personality Prediction Fixing"):
                idx, response, flag = future.result()
                if flag is False:
                    results[idx] = response
                else:
                    new_error_list.append([idx, task_list[idx]])
        error_list = new_error_list
    if output_path is not None:
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump({'preds': results, 'labels': test_sample_library['labels']}, f)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump({'preds': results, 'labels': test_sample_library['labels']}, f)
    return results
