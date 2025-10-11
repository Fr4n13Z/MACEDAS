import re
from typing import List, Dict, Optional

import emoji
import json5
import requests
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import extract_code_from_python


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

def writer_agent_request(response: requests.models.Response):
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
        posts = result.split("</think>")[-1].strip().replace("[ENDTHINKFLAG]", "")
        if 'action_input' in posts:
            posts = eval(posts)
            posts = posts['action_input'].strip()
        if "```python" in posts:
            posts = extract_code_from_python(posts).strip()
        posts = eval(posts)
    except Exception as e:
        flag = False
        posts = result.split("</think>")[-1].strip()
    return think_content, posts, flag


def writer_dify_api_call(segment_id: int | str, text_type: str, platform_name: str, input_text: str, api_key: str, base_url: str):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        "inputs": {'text_type': text_type, "platform_name": platform_name},
        "query": f"{input_text}",
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "anonymous"
    }

    response = requests.post(base_url, headers=headers, data=json.dumps(data))
    think_content, segments, flag = writer_agent_request(response)
    return segment_id, input_text, think_content, segments, flag


def multiple_thread_writer_api_call(user_texts: List[str],
                                  app_key: str,
                                  app_url: str,
                                  correction_try: bool = False,
                                  max_workers: int = 12,
                                  description: str = "",
                                  text_type: str="posts",
                                  platform_name: str="Facebook"):
    correct_list = [[] for _ in range(len(user_texts))]
    error_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as main_executor:

        futures = [main_executor.submit(writer_dify_api_call, idx, text_type, platform_name, user_text, app_key, app_url) for idx, user_text in
                   enumerate(user_texts)]

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

                futures = [fix_executor.submit(writer_dify_api_call, item[0], text_type, platform_name, item[1], app_key, app_url) for item in
                           error_list]

                for future in tqdm(as_completed(futures), total=len(futures), desc=description + " Fixing Error"):
                    idx, user_text, think_cot, segments, flag = future.result()
                    if flag:
                        correct_list[idx] = [idx, user_text, think_cot, segments]
                    else:
                        new_error_list.append([idx, user_text, think_cot, segments])
            error_list = new_error_list
    return correct_list, error_list


def correction_thread_writer_api_call(correct_list: List[List[str]],
                                    error_list: List[List[str]],
                                    app_key: str,
                                    app_url: str,
                                    max_workers: int = 12,
                                    description: str = "",
                                    text_type: str="posts",
                                    platform_name: str="Facebook"):
    while error_list:
        new_error_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as fix_executor:

            futures = [fix_executor.submit(writer_dify_api_call, text_type, platform_name, item[0], item[1], app_key, app_url) for item in
                       error_list]

            for future in tqdm(as_completed(futures), total=len(futures), desc=description):
                idx, user_text, think_cot, segments, flag = future.result()
                if flag:
                    correct_list[idx] = [idx, user_text, think_cot, segments]
                else:
                    new_error_list.append([idx, user_text, think_cot, segments])
        error_list = new_error_list
    return correct_list