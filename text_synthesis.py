import argparse
import json
import pickle
from sentence_transformers import SentenceTransformer
import os

from Synthesis import (construct_ref_facebook_library,
                       construct_ref_reddit_library,
                       personality_driven_user_match,
                       semantic_driven_text_match,
                       multiple_thread_writer_api_call,
                       judge_trait)


def reference_example_selection(encode_model, referenced_data, aligned_data, platform_type):

    input_labels = [{'mask': label, 'features': align_label} for label, align_label in
                    zip(aligned_data['labels'], aligned_data['aligned_labels'])]
    if platform_type == 'Facebook':
        reference_labels = [{'mask': label, 'features': [i / 7.0 for i in label_dist]} for label, label_dist in
                            zip(referenced_data['labels'], referenced_data['label_dists'])]
    elif platform_type == 'Reddit':
        reference_labels = [{'mask': label, 'features': [i for i in label_dist]} for label, label_dist in
                            zip(referenced_data['labels'], referenced_data['label_dists'])]
    else:
        raise ValueError("Invalid platform type")

    print(">>> Step 2.1: Personality Driven User Matching")
    pair_results = personality_driven_user_match(input_labels, reference_labels)

    input_samples = [[item['Segment'] for item in items if item['Segment'] != ''] for items in aligned_data['segments']]
    referenced_samples = [referenced_data['text'][item[0]][:100] for item in pair_results]

    print(">>> Step 2.2: Semantics based Post Matching")
    match_text_results = semantic_driven_text_match(encode_model, input_samples, referenced_samples, k=3, batch_size=16)
    referenced_language_styles = [referenced_data['language_style_analysis'][item[0]] for item in pair_results]
    selected_referenced_texts = [[[text[2] for text in item] for item in items] for items in match_text_results]

    aligned_data['referenced_language_style'] = referenced_language_styles
    aligned_data['referenced_posts'] = selected_referenced_texts

    input_personality_trends = [
        f"{judge_trait(new_label['features'][0])} Extraversion, {judge_trait(new_label['features'][1])} Neuroticism, {judge_trait(new_label['features'][2])} Agreeableness, {judge_trait(new_label['features'][3])} Conscientiousness, {judge_trait(new_label['features'][4])} Openness"
        for new_label in input_labels]
    aligned_data['trends'] = input_personality_trends
    return aligned_data


def text_generation_preprocess(matched_data):

    input_samples = [[item['Segment'] for item in items if item['Segment'] != ''] for items in matched_data['segments']]
    input_summaries = [[item['Summary'] for item in items if item['Segment'] != ''] for items in
                       matched_data['segments']]
    referenced_language_styles = matched_data['referenced_language_style']
    selected_referenced_texts = matched_data['referenced_posts']
    input_personality_trends = matched_data['trends']

    id_list = []
    input_task_list = []

    for idx, items in enumerate(
            zip(input_samples, input_summaries, referenced_language_styles, selected_referenced_texts,
                input_personality_trends)):
        i_segments, i_summaries, lang_style, ref_posts, trend = items
        for jdx, item in enumerate(zip(i_segments, i_summaries, ref_posts)):
            i_segment, i_summary, ref_post = item
            json_data = {'Simulated Personality': trend, 'Self Report': i_segment, 'Mind Summary': i_summary,
                         'Referenced Posts': ref_post, 'Referenced Style': lang_style}
            id_list.append([idx, jdx])
            input_task_list.append(json.dumps(json_data))

    return id_list, input_task_list


def parse_args():
    parser = argparse.ArgumentParser()
    # Encoder Model Path
    parser.add_argument('--encode_model_path', type=str, default="./PTM/jina-embedding-v3")

    # local LLM API
    parser.add_argument('--model', type=str, default='QwQ-32B', help='Local LLM name')
    parser.add_argument('--llm_url', type=str, default='http://192.168.1.1:8080/v1', help='Local LLM API URL')
    parser.add_argument("--api_key", type=str, default="EMPTY", help="Local LLM API key")

    # Dify Agent API
    parser.add_argument("--dify_app_url", type=str, default="http://localhost:80/v1/chat-messages", help='Dify Writer Agent API URL')
    parser.add_argument("--writer_key", type=str, default='app-xxx', help='Dify Writer Agent API key')

    # Data Path
    parser.add_argument("--input_path", type=str, default="./result/QwQ_32B/final_aligned_essays.json")   # "./data/OSData/Benchmark_Essays_original.pkl")
    parser.add_argument('--reference_path', type=str, default="./data/Benchmark_MyPersonality_original.pkl")
    parser.add_argument('--ref_output_path', type=str, default="./result/transfer_library/facebook_reference.json")
    parser.add_argument('--output_path', type=str, default="./result/QwQ_32B/final_gen_texts_facebook.pkl")

    # Target Generation Text Type & Platform
    parser.add_argument('--text_type', type=str, default='shared posts')
    parser.add_argument('--platform_name', type=str, default='Facebook')
    return parser.parse_args()


def personality_driven_text_generation(args: argparse.Namespace):
    with open(args.input_path, 'r') as f:
        aligned_data = json.load(f)
    print("----------------------------------")
    print(">>> Step 1: Language Style Analysis & Reference Library Construction")
    referenced_data = construct_ref_facebook_library(data_path=args.reference_path,
                                          model=args.model,
                                          base_url=args.llm_url,
                                          api_key=args.api_key,
                                          output_path=args.ref_output_path)
    encode_model = SentenceTransformer(args.encode_model_path, trust_remote_code=True, local_files_only=True)
    print("----------------------------------")
    print(">>> Step 2: Reference Library Retrieval")
    matched_data = reference_example_selection(encode_model=encode_model,
                                               referenced_data=referenced_data,
                                               aligned_data=aligned_data,
                                               platform_type=args.platform_name)
    id_list, input_task_list = text_generation_preprocess(matched_data)
    print("----------------------------------")
    print(">>> Step 3: LLM-based Text Generation")
    results, error_list = multiple_thread_writer_api_call(input_task_list,
                                                          args.writer_key,
                                                          args.dify_app_url,
                                                          correction_try=True,
                                                          text_type=args.text_type,
                                                          platform_name=args.platform_name)
    user_generated_texts = [[] for _ in range(len(aligned_data['segments']))]
    for sid, item in enumerate(id_list):
        idx, jdx = item
        user_generated_texts[idx].insert(jdx, results[sid])

    aligned_data['synthesized_texts_all'] = user_generated_texts
    aligned_data['synthesized_texts'] = [[item[-1] for item in user_generated_texts[i]] for i in range(len(user_generated_texts))]
    if args.output_path is not None:
        if args.output_path.endswith('.json'):
            with open(args.output_path, 'w') as f:
                json.dump(aligned_data, f, indent=4, ensure_ascii=False)
        else:
            with open(args.output_path, 'wb') as f:
                pickle.dump(aligned_data, f)
    return user_generated_texts


if __name__ == '__main__':
    args = parse_args()
    personality_driven_text_generation(args)