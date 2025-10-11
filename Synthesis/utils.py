import json
import re
from typing import Optional, List, Dict

import emoji
import requests


def remove_chinese(text):

    pattern = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')
    return pattern.sub('', text)


def text_preprocessing(text: str, lower: bool = True):
    text = emoji.demojize(text)
    if lower:
        text = text.lower()
    text = remove_chinese(text)
    return text


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
        "temperature": 0.7,
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



def extract_code_from_plain(content):
    pattern = r'```plain\s*(.*?)```'
    result = re.search(pattern, content.strip(), re.DOTALL)

    if result:
        plain_text = result.group(1).strip()
    else:
        plain_text = content.replace("```plain", "").replace("```", "").strip()
        print("[WARNING] No plain text found.")
    return plain_text


def extract_code_from_python(content):
    pattern = r'```python\s*(.*?)```'
    result = re.search(pattern, content.strip(), re.DOTALL)

    if result:
        extracted_code = result.group(1).strip()
    else:
        extracted_code = content.strip()
        print("[WARNING] No python code found.")
    return extracted_code