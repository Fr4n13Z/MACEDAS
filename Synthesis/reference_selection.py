import torch
import numpy as np
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def mask_similarity(mask1, mask2):
    intersection = sum(m1 == m2 for m1, m2 in zip(mask1, mask2))
    union = len(mask1)
    return intersection / union


def personality_driven_user_match(samples, references):
    mask_to_refs = defaultdict(list)
    for ref_idx, ref in enumerate(references):
        mask_tuple = tuple(ref['mask'])
        mask_to_refs[mask_tuple].append((ref_idx, ref['features']))

    results = []

    for sample in tqdm(samples, desc="Matching users"):
        sample_mask = tuple(sample['mask'])
        sample_features = np.array(sample['features'])

        if sample_mask in mask_to_refs:
            matched_refs = mask_to_refs[sample_mask]
            best_ref_idx, max_similarity = -1, -1

            for ref_idx, ref_features in matched_refs:
                ref_features = np.array(ref_features)
                similarity = np.dot(sample_features, ref_features) / (
                        np.linalg.norm(sample_features) * np.linalg.norm(ref_features))

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_ref_idx = ref_idx

            results.append((best_ref_idx, 'exact_mask', max_similarity))
            continue


        max_mask_sim = -1
        candidates = []

        for ref_mask, ref_data in mask_to_refs.items():
            sim = mask_similarity(sample_mask, ref_mask)
            if sim >= 0.6:  # 相似度阈值
                if sim > max_mask_sim:
                    max_mask_sim = sim
                    candidates = [(ref_idx, feats) for ref_idx, feats in ref_data]
                elif sim == max_mask_sim:
                    candidates.extend([(ref_idx, feats) for ref_idx, feats in ref_data])

        if not candidates:
            results.append((-1, 'no_match', 0))
            continue

        best_ref_idx, max_similarity = -1, -1
        for ref_idx, ref_features in candidates:
            ref_features = np.array(ref_features)
            similarity = np.dot(sample_features, ref_features) / (
                    np.linalg.norm(sample_features) * np.linalg.norm(ref_features))

            if similarity > max_similarity:
                max_similarity = similarity
                best_ref_idx = ref_idx

        results.append((best_ref_idx, 'similar_mask', max_similarity))

    return results


def semantic_driven_text_match(model, input_samples, reference_samples, k=3, batch_size=32):
    assert len(input_samples) == len(reference_samples), "Please ensure that the input and reference samples have the same length."

    ref_embeddings_list = []
    for ref_texts in tqdm(reference_samples, desc="Preprocessing reference samples"):
        if not ref_texts:
            ref_embeddings_list.append(torch.tensor([]))
            continue

        embeddings = []
        for i in range(0, len(ref_texts), batch_size):
            batch = ref_texts[i:i + batch_size]
            batch_embeds = model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeds.to(model.device))

        ref_embeds = torch.cat(embeddings, dim=0) if embeddings else torch.tensor([]).to(model.device)
        ref_embeddings_list.append(ref_embeds)

    results = []

    for input_idx, input_texts in enumerate(tqdm(input_samples, desc="matching input samples")):

        ref_embeds = ref_embeddings_list[input_idx]
        ref_texts = reference_samples[input_idx]

        if not input_texts:
            results.append([])
            continue

        input_embeds = []
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i + batch_size]
            batch_embeds = model.encode(batch, convert_to_tensor=True)
            input_embeds.append(batch_embeds.to(model.device))

        input_embeds = torch.cat(input_embeds, dim=0) if input_embeds else torch.tensor([]).to(model.device)

        if input_embeds.shape[0] == 0 or ref_embeds.shape[0] == 0:
            sim_matrix = np.zeros((len(input_texts), len(ref_texts)))
        else:
            sim_matrix = cosine_similarity(
                input_embeds.detach().to(torch.float).cpu().numpy(),
                ref_embeds.detach().to(torch.float).cpu().numpy()
            )

        sample_results = []
        for text_idx, text in enumerate(input_texts):
            if text_idx >= sim_matrix.shape[0]:
                sample_results.append([])
                continue

            sim_scores = sim_matrix[text_idx]
            valid_indices = np.where(~np.isnan(sim_scores))[0]

            if len(valid_indices) == 0 or len(ref_texts) == 0:
                sample_results.append([])
                continue

            top_k_indices = np.argsort(sim_scores[valid_indices])[-k:][::-1]
            top_k_indices = valid_indices[top_k_indices]

            matches = []
            for idx in top_k_indices:
                if idx >= len(ref_texts):
                    continue
                matches.append((
                    input_idx,
                    idx,
                    ref_texts[idx].replace("...", ""),
                    float(sim_scores[idx])
                ))

            sample_results.append(matches)

        results.append(sample_results)

    return results