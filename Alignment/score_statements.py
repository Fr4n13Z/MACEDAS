import argparse
import copy
import json
import pickle
import re
from typing import Optional, List, Dict

import json5
import openai
import pandas as pd
from tqdm import tqdm

from .capture_facets import facet_correction
from .utils import (multiple_thread_dify_api_call,
                    local_llm_request,
                    extract_code_from_python)


rating_correction_prompt = """
**Role**: You are a professional linguist in rating type correction.

**Task**: Please map the given words from the input_list into the corresponding ratings types in the `rating set`, your results is only the python dictionary.

**Requirements**:
1. The result should be a python dictionary, the keys are the words from input_list, and the values are the corresponding words from List, without any explanation or extra information.
2. The values should only selected in `rating set`, each key is only one best match word from `rating set`, not all possible matches.
3. If the given word is the number, the number bigger, the agreement trend is higher (normally, `1` is `Disagree`, `5` is `Agree`).
4. If the given word is uncertain in rating judgment, please use `Neutral` as the rating type.

**long-term memory**:
rating set is:
```python
rating_set = ['Disagree', 'Partially Disagree', 'Neutral', 'Partially Agree', 'Agree']
```

-----

**Example for Clarify**:

Input:
```python
input_list = [1, 2, 3, 4, 5, "1", "2", "3", "4", "5", "Disagree", "Disallow", "D isagree", "Disgree", "DISAGREE", "Partial disagreement", "Partial Disagree", "Partial Disgree", "PartialDisagree", "PartiallDisagree", "Neutra", "Neutr al", "Indifferent", "Neutr Neutral", "Neu", "Neural", "Partially Agree", "Partally Agree", "Partial agree", "Partial y Agree", "Partiall Agree", "Partialley Agree", "Partially Agre", "Parti ally Agree", "Partiallly Agree", "Partial ly Agree", " Partially Agree", "Partialagr Agree", "Am not easily disturbed by events.", "Adapt easily to new situations.", "Neither Agree nor Disagree", "Partial']

Output:
```python
{"1": "Disagree", "2": "Partially Disagree", "3": "Neutral", "4": "Partially Agree", "5": "Agree", "1": "Disagree", "2": "Partially Disagree", "3": "Neutral", "4": "Partially Agree", "5": "Agree", "Disagree": "Disagree", "Disallow": "Disagree", "D isagree": "Disagree", "Disgree": "Disagree", "DISAGREE": "Disagree", "Partial disagreement": "Partially Disagree", "Partial Disagree": "Partially Disagree", "Partial Disgree": "Partially Disagree", "PartialDisagree": "Partially Disagree", "PartiallDisagree": "Partially Disagree", "Neutra": "Neutral", "Neutr al": "Neutral", "Indifferent": "Neutral", "Neutr Neutral": "Neutral", "Neu": "Neutral", "Neural": "Neutral", "Partially Agree": "Partially Agree", "Partally Agree": "Partially Agree", "Partial agree": "Partially Agree", "Partial y Agree": "Partially Agree", "Partiall Agree": "Partially Agree", "Partialley Agree": "Partially Agree", "Partially Agre": "Partially Agree", "Parti ally Agree": "Partially Agree", "Partiallly Agree": "Partially Agree", "Partial ly Agree": "Partially Agree", " Partially Agree": "Partially Agree", "Partialagr Agree": "Partially Agree", "Am not easily disturbed by events.": "Agree", "Adapt easily to new situations.": "Agree", "Neither Agree nor Disagree": "Neutral", "Partial": "Neutral"}
```
"""


def rate_correction(api_key: str, model: str, url: str, generated_rates: List[str]):
    client = openai.OpenAI(api_key=api_key, base_url=url)
    messages = [{'role': 'system', 'content': rating_correction_prompt},
                {'role':'user', 'content': f'input_list={generated_rates}'}]
    response = client.chat.completions.create(model=model,
                                              messages=messages)  # local_llm_request(model, messages=messages, base_url=url)
    response = response.choices[0].message.content
    if "```python" in response:
        response = extract_code_from_python(response)
    return json5.loads(response)



def score_statements_via_analyst(args: argparse.Namespace,
                                 data: dict | Optional[str],
                                 item_path: str,
                                 app_key: str,
                                 app_url: str,
                                 output_path: Optional[str]=None,
                                 process_type: str='local'):
    # load item data
    item_df = pd.read_csv(item_path)
    item_df['trait'] = item_df['Key&Num.'].apply(lambda x: x[0])
    item_df['Key'] = item_df['Key'].apply(lambda x: x.lower())

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
        raise ValueError("Invalid process_type")
    facet_list = data['facets']

    # construct the input rating list
    rating_list = []
    id_list = []
    for uid, item in tqdm(enumerate(zip(segment_list, facet_list)), desc='Constructing Rating Input'):
        segments, facets = item
        facets_res = facets[-1]
        assert len(segments) == len(facets_res)
        for fid in range(len(facets[-1])):
            if '' in facets_res[fid]['facets']:
                facets_res[fid]['facets'].remove('')
            segments[fid]['facets'] = [facet.strip().lower() for facet in facets_res[fid]['facets']]
            rating_list.append(segments[fid])
            id_list.append([uid, fid])
    sign_list = [{} for i in range(len(rating_list))]

    # add the statements to the rating list
    for uuid, item in enumerate(rating_list):
        item_statements = {}
        for fuuid, facet in enumerate(item['facets']):
            ijk_statements = copy.deepcopy(item_df[item_df['Key'] == facet])
            item_statements[facet] = ijk_statements[ijk_statements['Sign'] == 1]['Item'].tolist()
            sign_list[uuid][facet] = [len(ijk_statements[ijk_statements['Sign'] == 1]['Item'].tolist())]
            item_statements[facet] += (ijk_statements[ijk_statements['Sign'] == -1]['Item'].tolist())
        rating_list[uuid]['facets'] = item_statements

    input_segments = [json.dumps(item) for item in rating_list]

    # Statement rating
    rated_response_list, _ = multiple_thread_dify_api_call(input_segments, app_key, app_url, correction_try=True, description='Rating Statement Process')

    facet_res_list = list({
        facet.lower()
        for item in rated_response_list
        for facet in item[-1].keys()
    })

    rated_res_list = list({
        value.lower()
        for value in rated_response_list
        for rates in value[-1].values()
        for value in rates
    })

    # Correct facets
    corrected_facet_res_dict = facet_correction(args.api_key, args.model, args.llm_url, facet_res_list)

    # Correct ratings
    corrected_rating_res_dict = rate_correction(args.api_key, args.model, args.llm_url, rated_res_list)

    # Update facets in place using direct iteration
    for idx in tqdm(range(len(rated_response_list)), desc='Updating Rating Results'):
        new_item_ratings = {}
        for key in rated_response_list[idx][-1].keys():
            corrected_key = corrected_facet_res_dict[key.lower()]
            new_item_ratings[corrected_key] = [corrected_rating_res_dict[value.lower()]
                                             for value in rated_response_list[idx][-1][key]]
        rated_response_list[idx][-1] = new_item_ratings

    # Rating result save
    ordered_scoring_list = [[] for _ in range(len(segment_list))]
    ordered_rating_list = [[] for _ in range(len(segment_list))]
    ordered_cot_list = [[] for _ in range(len(segment_list))]
    for ids, item in zip(id_list, rated_response_list):
        ordered_scoring_list[ids[0]].insert(ids[1], item[-1])
        ordered_cot_list[ids[0]].insert(ids[1], item[-2])
        ordered_rating_list[ids[0]].insert(ids[1], json.loads(item[1]))

    data[f'{process_type}_indicator'] = ordered_rating_list
    data[f'{process_type}_scoring'] = ordered_scoring_list
    data[f'{process_type}_scoring_cot'] = ordered_cot_list

    if output_path is not None:
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
    return data




