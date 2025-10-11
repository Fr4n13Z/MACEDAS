import argparse
import json
import pickle
from typing import Optional
import openai
import json5
from tqdm import tqdm

from .utils import (multiple_thread_dify_api_call,
                    correction_thread_dify_api_call,
                    extract_code_from_python)

facet_correction_prompt = """
**Role**: You are a professional linguist in psychological facets.

**Task**: Please map the given words from the input_list into the corresponding facets in the `facet set`, your results is only the python dictionary.

**Requirements**:
1. The result should be a python dictionary, the keys are the words from input_list, and the values are the corresponding words from List, without any explanation or extra information.
2. The values should only selected in `facet set`, each key is only one best match word from `facet set`, not all possible matches.

**long-term memory**:
facet set is:
```python
facet_set = ['anxiety', 'anger', 'depression', 'self-consciousness', 'immoderation', 'vulnerability', 'friendliness', 'gregariousness', 'assertiveness', 'activity level', 'excitement-seeking', 'cheerfulness', 'imagination', 'artistic interests', 'emotionality', 'adventurousness', 'intellect', 'liberalism', 'trust', 'morality', 'altruism', 'cooperation', 'modesty', 'sympathy', 'self-efficacy', 'orderliness', 'dutifulness', 'achievement-striving', 'self-discipline', 'cautiousness']
```

-----

**Example for Clarify**:

Input:
```python
input_list = ['Intellect', 'Gregariousness', 'Self-Discipline', 'SelfDiscipline', 'Cooperation', 'Orderliness', 'Self_Efficacy']
```

Output:
```python
{'Intellect': 'intellect', 'Gregariousness': 'gregariousness', 'Self-Discipline': 'self-discipline', 'SelfDiscipline': 'self-discipline', 'Cooperation': 'cooperation', 'Orderliness': 'orderliness', 'Self_Efficacy': 'self-efficacy'}
```
"""


def facet_correction(api_key: str, model: str, url: str, generated_facets: list):
    client = openai.OpenAI(api_key=api_key, base_url=url)
    messages = [{'role': 'system', 'content': facet_correction_prompt},
                {'role':'user', 'content': f'input_list={generated_facets}'}]
    response = client.chat.completions.create(model=model, messages=messages)   # local_llm_request(model, messages=messages, base_url=url)
    response = response.choices[0].message.content
    if "```python" in response:
        response = extract_code_from_python(response)
    return json5.loads(response)


def error_length_check(segment_list, captured_response_list):
    error_list = []
    # check the length of user segments
    for segments, facets in zip(segment_list, captured_response_list):
        if len(segments) != len(facets[-1]):
            error_list.append(facets)
    return error_list

def capture_facets_via_psychologist(args: argparse.Namespace,
                                    data: dict | Optional[str],
                                    app_key: str,
                                    app_url: str,
                                    output_path: Optional[str]=None,
                                    process_type: str='local',
                                    facet_valid: bool=True):
    if isinstance(data, str):
        # Input data is file, load data
        if data.endswith('.json'):
            with open(data, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(data, 'rb') as f:
                data = pickle.load(f)

    # data preprocess
    if process_type == 'local':
        segment_list = [item[-1] for item in data['segments']]
    elif process_type == 'global':
        text_list = [text.replace("\"", '\'').strip() for text in data["text"]]
        segment_list = [[{'Summary': "", 'Segment': text}] for text in text_list]
    else:
        raise ValueError("Invalid process_type. Please choose 'local' or 'global'.")
    input_segments = [json.dumps(segment) for segment in segment_list]

    # Facet capture
    captured_response_list, _ = multiple_thread_dify_api_call(input_segments[:10], app_key, app_url, correction_try=True, description="Capturing Facet Process")

    # check the length of user segments is equal to the length of facet results
    error_list = error_length_check(segment_list, captured_response_list)
    while error_list:
        captured_response_list = correction_thread_dify_api_call(captured_response_list, error_list, app_key, app_url, description="Correcting Error Length Process")
        error_list = error_length_check(segment_list, captured_response_list)

    if process_type == 'local':
        # Collect all unique facets first using list comprehension and set
        facet_res_list = list({
            facet.lower()
            for item in captured_response_list
            for seg_res in item[-1]
            for facet in seg_res['facets']
        })
    else:
        for idx in tqdm(range(len(captured_response_list)), desc="Updating Facet Results"):
            if isinstance(captured_response_list[idx][-1], dict):
                captured_response_list[idx][-1] = [captured_response_list[idx][-1]]
            if 'facets' not in captured_response_list[idx][-1][0].keys():
                captured_response_list[idx][-1] = [{'facets': list(captured_response_list[idx][-1][0].values())[0]}]
        facet_res_list = list({
            facet.lower()
            for item in captured_response_list
            for seg_res in item[-1]
            for facet in seg_res['facets']
        })
    # Correct facets
    corrected_facet_res_dict = facet_correction(args.api_key, args.model, args.llm_url, facet_res_list)

    # Update facets in place using direct iteration
    for idx in tqdm(range(len(captured_response_list)), desc="Updating Facet Results"):
        for jdx in range(len(captured_response_list[idx][-1])):
            for kdx, key in enumerate(captured_response_list[idx][-1][jdx]['facets']):
                captured_response_list[idx][-1][jdx]['facets'][kdx] = corrected_facet_res_dict[key.lower()]

    # Facet result save
    data['facets'] = captured_response_list
    if output_path is not None:
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
    return data



