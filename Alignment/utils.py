import re
from typing import List, Dict, Optional

import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def dify_agent_request(response: requests.models.Response):
    result = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                json_data = json.loads(decoded_line.replace("data:", "").strip())
                if 'answer' in json_data:
                    result += json_data['answer']
    think_content = result.split("</think>")[0].strip().replace("<think>", "")
    flag = True
    try:
        segments = result.split("</think>")[-1].strip()
        if "```json" in segments:
            segments = extract_code_from_json(segments).strip()
        segments = json.loads(segments)
        if 'action_input' in segments:
            segments = segments['action_input']
            if isinstance(segments, str):
                segments = json.loads(segments.strip())
    except Exception as e:
        flag = False
        segments = result.split("</think>")[-1].strip()
    return think_content, segments, flag


def dify_api_call(segment_id: int | str, input_text: str, api_key: str, base_url: str):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        "inputs": {},
        "query": f"Author's text: {input_text}",
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "anonymous"
    }

    response = requests.post(base_url, headers=headers, data=json.dumps(data))
    think_content, segments, flag = dify_agent_request(response)
    return segment_id, input_text, think_content, segments, flag


"""
def process_dify_api(segment_id, input_text, api_key, base_url):
    segment_id, input_text, think_content, segments, flag = dify_api_call(segment_id, input_text, api_key, base_url)
    return segment_id, input_text, think_content, segments, flag"""


def multiple_thread_dify_api_call(user_texts: List[str],
                                  app_key: str,
                                  app_url: str,
                                  correction_try: bool = False,
                                  max_workers: int = 12,
                                  description: str = ""):
    correct_list = [[] for _ in range(len(user_texts))]
    error_list = []
    # 多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as main_executor:
        # 提交所有任务
        futures = [main_executor.submit(dify_api_call, idx, user_text, app_key, app_url) for idx, user_text in
                   enumerate(user_texts)]
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(futures), desc=description):
            idx, user_text, think_cot, segments, flag = future.result()
            if flag:
                correct_list[idx] = [idx, user_text, think_cot, segments]
            else:
                error_list.append([idx, user_text, think_cot, segments])
    if correction_try:
        while error_list:
            new_error_list = []
            with ThreadPoolExecutor(max_workers=max_workers) as fix_executor:
                # 提交所有任务
                futures = [fix_executor.submit(dify_api_call, item[0], item[1], app_key, app_url) for item in
                           error_list]
                # 使用tqdm显示进度
                for future in tqdm(as_completed(futures), total=len(futures), desc=description + " Fixing Error"):
                    idx, user_text, think_cot, segments, flag = future.result()
                    if flag:
                        correct_list[idx] = [idx, user_text, think_cot, segments]
                    else:
                        new_error_list.append([idx, user_text, think_cot, segments])
            error_list = new_error_list
    return correct_list, error_list


def correction_thread_dify_api_call(correct_list: List[List[str]],
                                    error_list: List[List[str]],
                                    app_key: str,
                                    app_url: str,
                                    max_workers: int = 12,
                                    description: str = ""):
    while error_list:
        new_error_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as fix_executor:
            # 提交所有任务
            futures = [fix_executor.submit(dify_api_call, item[0], item[1], app_key, app_url) for item in
                       error_list]
            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(futures), desc=description):
                idx, user_text, think_cot, segments, flag = future.result()
                if flag:
                    correct_list[idx] = [idx, user_text, think_cot, segments]
                else:
                    new_error_list.append([idx, user_text, think_cot, segments])
        error_list = new_error_list
    return correct_list


def local_llm_request(model: str,
                      messages: List[Dict[str, str]],
                      base_url: str,
                      verify: Optional[bool] = False,
                      timeout: Optional[int] = None,
                      stream: Optional[bool] = False):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer EMPTY'
    }
    if "/v1/chat/completions" not in base_url:
        base_url = base_url + "/v1/chat/completions"
    payload_dict = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": 0.,
    }
    payload = json.dumps(payload_dict, ensure_ascii=False)
    result_payload = requests.post(url=base_url,
                                   data=payload.encode('utf-8'),
                                   headers=headers,
                                   verify=verify,
                                   timeout=timeout)
    response_dict = json.loads(result_payload.text)
    if 'choices' not in response_dict.keys():
        return f"Error Response: {response_dict}"
    generated_text = response_dict['choices'][0]['message']['content'].strip()
    if "</think>" in generated_text:
        generated_text = generated_text.split("</think>")[-1].strip()
    return generated_text


facet_correlation_dict = {
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


facet_list = list(facet_correlation_dict.keys())


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