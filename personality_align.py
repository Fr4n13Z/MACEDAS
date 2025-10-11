import argparse
import json
import pickle

from Alignment import (split_segments_via_linguist,
                       capture_facets_via_psychologist,
                       score_statements_via_analyst,
                       align_labels_via_global_and_local_views)

def parse_args():
    parser = argparse.ArgumentParser()

    # local LLM API
    parser.add_argument('--model', type=str, default='QwQ-32B')
    parser.add_argument('--llm_url', type=str, default='http://192.168.1.1:8080/v1',help="Your local LLM API URL")

    # Dify Agent API
    parser.add_argument("--dify_app_url", type=str, default="http://localhost:80/v1/chat-messages", help='Your Dify Agent API URL')
    parser.add_argument("--api_key", type=str, default="EMPTY", help='Your LLM API Key')
    parser.add_argument("--linguist_key", type=str, default='app-xxx', help="Your Dify Linguist Agent API Key")
    parser.add_argument("--psychologist_key", type=str, default='app-xxx', help="Your Dify Psychologist Agent API Key")
    parser.add_argument("--analyst_key", type=str, default='app-xxx', help="Your Dify Analyst Agent API Key")

    # Data Path+
    parser.add_argument("--input_path", type=str, default="./data/Benchmark_Essays_original.pkl")  # "./data/OSData/Benchmark_Essays_original.pkl") # "./data/youtube/data.pkl")
    parser.add_argument('--item_path', type=str, default="./data/items/IPIP-NEO-300Items.csv")
    parser.add_argument('--output_path', type=str, default="./result/QwQ_32B/{step}_alignment.json")
    return parser.parse_args()


def local_scoring_process(args: argparse.Namespace):
    print("-----------------------------------------")
    print("Local scoring process:")
    # Step 1: Split segments via Linguist
    print("[Step 1] >>> Split segments via Linguist")
    data = split_segments_via_linguist(args.input_path, args.linguist_key, args.dify_app_url, args.output_path.format(step=f'{args.model}_l1'))
    # Step 2: Capture facets via Psychologist
    print("[Step 2] >>> Psychological Facet Extraction via Psychologist")
    data = capture_facets_via_psychologist(args, data, args.psychologist_key, args.dify_app_url, args.output_path.format(step=f'{args.model}_l2'))
    # Step 3: Score statements via Analyst
    print("[Step 3] >>> Facet Statement Scoring via Assessment Analyst")
    data = score_statements_via_analyst(args, data, args.item_path, args.analyst_key, args.dify_app_url, args.output_path.format(step=f'{args.model}_l3'))
    return data


def global_scoring_process(args: argparse.Namespace):
    print("-----------------------------------------")
    print("Global scoring process:")
    # Step 1: Capture facets via Psychologist
    print("[Step 1] >>> Psychological Facet Extraction via Psychologist")
    data = capture_facets_via_psychologist(args, args.input_path, args.psychologist_key, args.dify_app_url, args.output_path.format(step=f'{args.model}_g1'), process_type='global')\
    # Step 2: Score statements via Analyst
    print("[Step 2] >>> Facet Statement Scoring via Assessment Analyst")
    data = score_statements_via_analyst(args, data, args.item_path, args.analyst_key, args.dify_app_url, args.output_path.format(step=f'{args.model}_g2'), process_type='global')
    return data


if __name__ == '__main__':
    args = parse_args()
    local_data = local_scoring_process(args)
    global_data = global_scoring_process(args)
    align_labels_via_global_and_local_views(args,
                                            local_data,
                                            global_data,
                                            args.item_path,
                                            args.output_path.format(step=f'{args.model}_final_aligned'))