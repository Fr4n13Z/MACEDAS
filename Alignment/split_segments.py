import json
import pickle
from typing import Optional

from .utils import multiple_thread_dify_api_call


def split_segments_via_linguist(data_path: str, app_key: str, app_url: str, output_path: Optional[str] = None):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    text_list = [text.replace("\"", '\'').strip() for text in data["text"]]
    split_response_list, _ = multiple_thread_dify_api_call(text_list, app_key, app_url, correction_try=True, description="Split Segments Process")
    data['segments'] = split_response_list
    if output_path is not None:
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
    return data