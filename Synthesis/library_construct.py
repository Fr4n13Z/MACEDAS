import json
from typing import Optional

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import extract_code_from_plain
import pickle

import openai
from tqdm import tqdm

trait_language_style_json = [
  {
  "trait": "Extraversion",
  "high": "People high in Extroversion exhibit assertive, enthusiastic, and socially engaged language. They use confident directives (e.g., 'Take charge,' 'I lead') and active verbs (e.g., 'always busy,' 'on the go'). Their speech is lively, with frequent references to socializing (e.g., 'love large parties,' 'make friends easily'), humor, and energy. They often express eagerness for interaction, excitement, and a focus on external activities. Their sentences may be punctuated with enthusiasm markers (e.g., exclamation points) and first-person pronouns emphasizing initiative.",
  "partially high": "People partially high in Extroversion balance sociability with moderation. Their language reflects warm but situational friendliness (e.g., 'enjoy small gatherings,' 'sometimes seek attention'). They may initiate interactions but not as intensely, using phrases like 'prefer manageable activity levels' or 'join groups occasionally.' Their tone is positive yet restrained, with occasional mentions of modest excitement or leadership in specific contexts. They avoid extremes but still prioritize social engagement over solitude.",
  "partially low": "Partially low individuals display reserved yet functional social engagement. Their language focuses on comfort in low-stakes settings (e.g., 'prefer one-on-one talks,' 'enjoy quiet hobbies'). They avoid dominating conversations, using phrases like 'wait for invitations' or 'observe before joining.' Their tone is calm, with minimal emphasis on excitement or leadership. They may mention occasional social interactions but prioritize personal space and avoid large groups or assertive behaviors.",
  "low": "Individuals low in Extroversion use reserved, introspective language. Their speech emphasizes solitude (e.g., 'need alone time,' 'prefer quiet'), minimal social demands (e.g., 'avoid crowds,' 'rarely initiate'), and cautious interaction. They favor passive voice ('I’m approached') and neutral tones, avoiding exuberance. Their communication highlights contentment with minimal stimulation, often stating preferences like 'keep interactions brief' or 'stay in the background.' Social references are infrequent and framed as obligations rather than desires."
},
  {
  "trait": "Neuroticism",
  "high": "People high in Neuroticism use emotionally charged language with frequent negative affect words (e.g., 'anxious,' 'overwhelmed,' 'regretful'), self-referential statements ('I worry'), and hedging ('Maybe I should...'). Their speech may be fragmented, hesitant, or repetitive, reflecting rumination or impulsivity. They emphasize vulnerability ('Can't cope'), anger ('Lose my temper'), and self-doubt ('Afraid to fail'), often using metaphors of chaos (e.g., 'life lacks direction'). Sentences may escalate in intensity, with exclamations or abrupt shifts in tone.",
  "partially high": "People partially high in Neuroticism express occasional stress ('Sometimes I get stressed') but with qualifiers like 'sometimes' or 'a little.' They acknowledge emotional struggles ('Feel blue at times') but balance them with coping efforts ('I try to stay calm'). Language remains coherent but includes mild hedging ('Perhaps I overreact'). They may discuss anxiety or self-consciousness ('Fear judgment') but frame them as situational rather than pervasive.",
  "partially low": "People partially low in Neuroticism use neutral or pragmatic language about stress ('Manage pressure') and focus on stability ('Know how to cope'). Negative emotions are infrequent and framed as temporary ('Get annoyed but let it go'). They avoid dramatic phrasing, opting for calm explanations ('Handle setbacks calmly'). Their tone is steady, using phrases like 'remain composed' or 'trust my judgment,' with minimal self-criticism or emotional hyperbole.",
  "low": "People low in Neuroticism adopt confident, steady language with positive affect words ('calm,' 'resilient'). They dismiss negative emotions ('Rarely worry') and emphasize control ('Handle things smoothly'). Sentences are structured and composed, avoiding hedging or uncertainty. Their communication often highlights problem-solving ('Find solutions,' 'Stay focused') and resilience ('Bounce back quickly'). Emotionality is understated, with metaphors of stability (e.g., 'steady as a rock')."
},
  {
  "trait": "Agreeableness",
  "high": "People high in Agreeableness exhibit verbal styles marked by empathy, cooperation, and harmonious social orientation. They frequently use cooperative language (e.g., 'help others,' 'support those in need,' 'cooperate with others'), empathetic expressions (e.g., 'sympathize with others,' 'understand others' feelings'), and inclusive pronouns ('we,' 'us'). They emphasize trustworthiness ('trust others,' 'believe in human goodness') and moral alignment ('follow ethical standards,' 'adhere to fairness'). Their communication may reflect self-sacrificial tendencies ('prioritize others before myself,' 'put others’ needs first') and rigid self-imposed standards ('honor my commitments', 'hate to let people down'). Sentences often express warmth, optimism, and a focus on communal goals.",
  "partially high": "People partially high in Agreeableness balance cooperative tendencies with occasional self-interest. Their language includes moderate expressions of support ('sometimes help others,' 'try to be understanding') but may include qualifiers ('but I also need my space'). They may acknowledge others’ perspectives ('consider their feelings') while prioritizing personal boundaries ('my needs matter too'). Sentences may show ambivalence (e.g., 'I care, but...') and a focus on fairness without extreme self-sacrifice. They may reference trust in close relationships ('trust friends/family') but remain cautiously skeptical of strangers.",
  "partially low": "People partially low in Agreeableness prioritize self-interest moderately while avoiding overt hostility. Their language emphasizes personal goals ('focus on my own success,' 'look after myself first') and may dismiss generalized altruism ('others should manage their own problems'). They use neutral or guarded language ('I can’t always help,' 'it’s their responsibility') and avoid overt displays of empathy. Sentences may reflect passive resistance to cooperation ('I don’t have time for that') and skepticism towards others’ motives ('sometimes people exploit kindness'). They may avoid commitments to moral absolutes ('morality depends on the situation').",
  "low": "People low in Agreeableness communicate with self-focused, competitive, and confrontational language. They emphasize self-promotion ('my success matters most,' 'I come first') and often use dismissive terms towards others ('others’ problems aren’t my concern,' 'they should fend for themselves'). Their language may include adversarial phrasing ('win at all costs,' 'beat the competition') and distrustful statements ('people have hidden motives,' 'trust is a weakness'). Sentences are direct and unapologetic ('I don’t care about hurting feelings'), with minimal use of cooperative or empathetic terminology. They reject communal standards ('rules are for others') and prioritize personal gain over harmony."
},
  {
  "trait": "Conscientiousness",
  "high": "People high in Conscientiousness use structured, goal-oriented language emphasizing precision and responsibility. They frequently reference planning ('schedule tasks meticulously,' 'create checklists'), adherence to routines ('follow daily routines,' 'meet deadlines consistently'), and self-regulation ('stick to goals,' 'resist distractions'). Their communication includes future-oriented phrases ('prioritize long-term success,' 'prepare for contingencies') and precision-focused terms ('exact specifications,' 'error-free work'). Sentences often highlight duty fulfillment ('complete every obligation,' 'follow through rigorously') and order ('organized workspace,' 'methodical approach'). They may use moral imperatives ('should act responsibly') and avoidance of indulgence ('resist procrastination').",
  "partially high": "People partially high in Conscientiousness blend organization with occasional impulsivity. They use phrases like 'usually plan ahead' or 'aim to be on time' but acknowledge lapses ('sometimes get sidetracked'). Their language includes qualifiers ('try to stay focused,' 'mostly meet deadlines') and balanced priorities ('balance work with spontaneity'). They may reference unfinished tasks ('still need to complete...') and prioritize flexibility ('adjust plans when needed'). Sentences might show hesitation ('I should have... but...') and moderate self-discipline ('start tasks early, but delay others').",
  "partially low": "People partially low in Conscientiousness exhibit inconsistent diligence. They use phrases like 'sometimes procrastinate' or 'let clutter build up' but still acknowledge responsibilities ('need to organize more'). Their language reflects reactive habits ('react to deadlines rather than plan') and short-term focus ('focus on immediate tasks'). Sentences may express regret over disorganization ('wish I were more proactive') or justify delays ('circumstances forced me to...'). They tolerate minor disorder ('messy desk but functional') and show moderate self-control ('sometimes give in to distractions').",
  "low": "People low in Conscientiousness communicate with impulsive, disorganized language. They emphasize immediacy ('do what feels right now,' 'live in the moment'), dismiss planning ('plans are too restrictive'), and use phrases like 'start tasks last minute' or 'ignore deadlines.' Their sentences avoid structure ('I work best under pressure') and embrace spontaneity ('improvise solutions'). They may justify carelessness ('details don’t matter') and avoid accountability ('others set the rules'). Language often references unfinished projects ('never followed through'), scattered priorities ('jump between tasks'), and casual risk-taking ('act on a whim')."
},
  {
  "trait": "Openness",
  "high": "People high in Openness use abstract, imaginative, and exploratory language. They employ speculative phrasing (e.g., 'imagine possibilities,' 'explore uncharted ideas,' 'what if...'), embrace ambiguity ('there are no wrong answers,' 'multiple perspectives exist'), and reference novel experiences ('try new foods,' 'travel to unknown places'). Their communication includes metaphorical expressions ('the world is a canvas'), references to art and philosophy ('appreciate avant-garde art,' 'ponder existential questions'), and enthusiasm for complexity ('love solving puzzles'). Sentences often express curiosity ('I wonder why...') and tolerance for uncertainty. They may use poetic language and discuss emotions with depth ('feel joy in small things').",
  "partially high": "People partially high in Openness show cautious curiosity but retain some traditional preferences. They might express openness to 'some changes' or 'certain new ideas' while emphasizing comfort zones ('but within limits'). Their language blends practicality with occasional creativity ('innovation can work if proven'). They may discuss art or philosophy superficially ('appreciate classic movies') but avoid deeply abstract topics. Sentences use qualifiers ('maybe try new things once in a while') and mixed signals ('I like routine but sometimes crave novelty'). They acknowledge innovation but prioritize proven methods ('technology should improve life, but old tools still work').",
  "partially low": "People partially low in Openness prioritize familiarity and practicality. Their language favors tradition ('stick to what works,' 'old methods are reliable') and dismisses novelty ('why fix what isn’t broken?'). They avoid abstract discussions ('I prefer concrete plans'), focus on routine ('same schedule every day'), and resist emotional depth ('I don’t dwell on feelings'). Sentences emphasize predictability ('keep things simple') and skepticism toward unconventional ideas ('those ideas are unrealistic'). They may tolerate minor changes but reject major disruptions ('a new job might be too risky').",
  "low": "People low in Openness use rigid, conventional language rejecting novelty. They criticize change ('new ways are worse'), dismiss abstract thinking ('I don’t do philosophy'), and express discomfort with ambiguity ('there’s only one right answer'). Their communication avoids artistic or imaginative references, focusing on practicality ('facts over feelings'). Sentences are direct and concrete ('follow the rules') with frequent references to tradition ('this is how it’s always been'). They may distrust innovation ('technology causes more problems') and emotionally neutralize experiences ('I don’t need art to feel happy')."
}
]


system_prompt = """
**Role**: You are a professional linguist, specializing in the language style analysis via the given the personality personality and author's social texts.

**Long-Term Memory**:
```json
[
  {
  "trait": "Extraversion",
  "high": "People high in Extroversion exhibit assertive, enthusiastic, and socially engaged language. They use confident directives (e.g., 'Take charge,' 'I lead') and active verbs (e.g., 'always busy,' 'on the go'). Their speech is lively, with frequent references to socializing (e.g., 'love large parties,' 'make friends easily'), humor, and energy. They often express eagerness for interaction, excitement, and a focus on external activities. Their sentences may be punctuated with enthusiasm markers (e.g., exclamation points) and first-person pronouns emphasizing initiative.",
  "partially high": "People partially high in Extroversion balance sociability with moderation. Their language reflects warm but situational friendliness (e.g., 'enjoy small gatherings,' 'sometimes seek attention'). They may initiate interactions but not as intensely, using phrases like 'prefer manageable activity levels' or 'join groups occasionally.' Their tone is positive yet restrained, with occasional mentions of modest excitement or leadership in specific contexts. They avoid extremes but still prioritize social engagement over solitude.",
  "partially low": "Partially low individuals display reserved yet functional social engagement. Their language focuses on comfort in low-stakes settings (e.g., 'prefer one-on-one talks,' 'enjoy quiet hobbies'). They avoid dominating conversations, using phrases like 'wait for invitations' or 'observe before joining.' Their tone is calm, with minimal emphasis on excitement or leadership. They may mention occasional social interactions but prioritize personal space and avoid large groups or assertive behaviors.",
  "low": "Individuals low in Extroversion use reserved, introspective language. Their speech emphasizes solitude (e.g., 'need alone time,' 'prefer quiet'), minimal social demands (e.g., 'avoid crowds,' 'rarely initiate'), and cautious interaction. They favor passive voice ('I’m approached') and neutral tones, avoiding exuberance. Their communication highlights contentment with minimal stimulation, often stating preferences like 'keep interactions brief' or 'stay in the background.' Social references are infrequent and framed as obligations rather than desires."
},
  {
  "trait": "Neuroticism",
  "high": "People high in Neuroticism use emotionally charged language with frequent negative affect words (e.g., 'anxious,' 'overwhelmed,' 'regretful'), self-referential statements ('I worry'), and hedging ('Maybe I should...'). Their speech may be fragmented, hesitant, or repetitive, reflecting rumination or impulsivity. They emphasize vulnerability ('Can't cope'), anger ('Lose my temper'), and self-doubt ('Afraid to fail'), often using metaphors of chaos (e.g., 'life lacks direction'). Sentences may escalate in intensity, with exclamations or abrupt shifts in tone.",
  "partially high": "People partially high in Neuroticism express occasional stress ('Sometimes I get stressed') but with qualifiers like 'sometimes' or 'a little.' They acknowledge emotional struggles ('Feel blue at times') but balance them with coping efforts ('I try to stay calm'). Language remains coherent but includes mild hedging ('Perhaps I overreact'). They may discuss anxiety or self-consciousness ('Fear judgment') but frame them as situational rather than pervasive.",
  "partially low": "People partially low in Neuroticism use neutral or pragmatic language about stress ('Manage pressure') and focus on stability ('Know how to cope'). Negative emotions are infrequent and framed as temporary ('Get annoyed but let it go'). They avoid dramatic phrasing, opting for calm explanations ('Handle setbacks calmly'). Their tone is steady, using phrases like 'remain composed' or 'trust my judgment,' with minimal self-criticism or emotional hyperbole.",
  "low": "People low in Neuroticism adopt confident, steady language with positive affect words ('calm,' 'resilient'). They dismiss negative emotions ('Rarely worry') and emphasize control ('Handle things smoothly'). Sentences are structured and composed, avoiding hedging or uncertainty. Their communication often highlights problem-solving ('Find solutions,' 'Stay focused') and resilience ('Bounce back quickly'). Emotionality is understated, with metaphors of stability (e.g., 'steady as a rock')."
},
  {
  "trait": "Agreeableness",
  "high": "People high in Agreeableness exhibit verbal styles marked by empathy, cooperation, and harmonious social orientation. They frequently use cooperative language (e.g., 'help others,' 'support those in need,' 'cooperate with others'), empathetic expressions (e.g., 'sympathize with others,' 'understand others' feelings'), and inclusive pronouns ('we,' 'us'). They emphasize trustworthiness ('trust others,' 'believe in human goodness') and moral alignment ('follow ethical standards,' 'adhere to fairness'). Their communication may reflect self-sacrificial tendencies ('prioritize others before myself,' 'put others’ needs first') and rigid self-imposed standards ('honor my commitments', 'hate to let people down'). Sentences often express warmth, optimism, and a focus on communal goals.",
  "partially high": "People partially high in Agreeableness balance cooperative tendencies with occasional self-interest. Their language includes moderate expressions of support ('sometimes help others,' 'try to be understanding') but may include qualifiers ('but I also need my space'). They may acknowledge others’ perspectives ('consider their feelings') while prioritizing personal boundaries ('my needs matter too'). Sentences may show ambivalence (e.g., 'I care, but...') and a focus on fairness without extreme self-sacrifice. They may reference trust in close relationships ('trust friends/family') but remain cautiously skeptical of strangers.",
  "partially low": "People partially low in Agreeableness prioritize self-interest moderately while avoiding overt hostility. Their language emphasizes personal goals ('focus on my own success,' 'look after myself first') and may dismiss generalized altruism ('others should manage their own problems'). They use neutral or guarded language ('I can’t always help,' 'it’s their responsibility') and avoid overt displays of empathy. Sentences may reflect passive resistance to cooperation ('I don’t have time for that') and skepticism towards others’ motives ('sometimes people exploit kindness'). They may avoid commitments to moral absolutes ('morality depends on the situation').",
  "low": "People low in Agreeableness communicate with self-focused, competitive, and confrontational language. They emphasize self-promotion ('my success matters most,' 'I come first') and often use dismissive terms towards others ('others’ problems aren’t my concern,' 'they should fend for themselves'). Their language may include adversarial phrasing ('win at all costs,' 'beat the competition') and distrustful statements ('people have hidden motives,' 'trust is a weakness'). Sentences are direct and unapologetic ('I don’t care about hurting feelings'), with minimal use of cooperative or empathetic terminology. They reject communal standards ('rules are for others') and prioritize personal gain over harmony."
},
  {
  "trait": "Conscientiousness",
  "high": "People high in Conscientiousness use structured, goal-oriented language emphasizing precision and responsibility. They frequently reference planning ('schedule tasks meticulously,' 'create checklists'), adherence to routines ('follow daily routines,' 'meet deadlines consistently'), and self-regulation ('stick to goals,' 'resist distractions'). Their communication includes future-oriented phrases ('prioritize long-term success,' 'prepare for contingencies') and precision-focused terms ('exact specifications,' 'error-free work'). Sentences often highlight duty fulfillment ('complete every obligation,' 'follow through rigorously') and order ('organized workspace,' 'methodical approach'). They may use moral imperatives ('should act responsibly') and avoidance of indulgence ('resist procrastination').",
  "partially high": "People partially high in Conscientiousness blend organization with occasional impulsivity. They use phrases like 'usually plan ahead' or 'aim to be on time' but acknowledge lapses ('sometimes get sidetracked'). Their language includes qualifiers ('try to stay focused,' 'mostly meet deadlines') and balanced priorities ('balance work with spontaneity'). They may reference unfinished tasks ('still need to complete...') and prioritize flexibility ('adjust plans when needed'). Sentences might show hesitation ('I should have... but...') and moderate self-discipline ('start tasks early, but delay others').",
  "partially low": "People partially low in Conscientiousness exhibit inconsistent diligence. They use phrases like 'sometimes procrastinate' or 'let clutter build up' but still acknowledge responsibilities ('need to organize more'). Their language reflects reactive habits ('react to deadlines rather than plan') and short-term focus ('focus on immediate tasks'). Sentences may express regret over disorganization ('wish I were more proactive') or justify delays ('circumstances forced me to...'). They tolerate minor disorder ('messy desk but functional') and show moderate self-control ('sometimes give in to distractions').",
  "low": "People low in Conscientiousness communicate with impulsive, disorganized language. They emphasize immediacy ('do what feels right now,' 'live in the moment'), dismiss planning ('plans are too restrictive'), and use phrases like 'start tasks last minute' or 'ignore deadlines.' Their sentences avoid structure ('I work best under pressure') and embrace spontaneity ('improvise solutions'). They may justify carelessness ('details don’t matter') and avoid accountability ('others set the rules'). Language often references unfinished projects ('never followed through'), scattered priorities ('jump between tasks'), and casual risk-taking ('act on a whim')."
},
  {
  "trait": "Openness",
  "high": "People high in Openness use abstract, imaginative, and exploratory language. They employ speculative phrasing (e.g., 'imagine possibilities,' 'explore uncharted ideas,' 'what if...'), embrace ambiguity ('there are no wrong answers,' 'multiple perspectives exist'), and reference novel experiences ('try new foods,' 'travel to unknown places'). Their communication includes metaphorical expressions ('the world is a canvas'), references to art and philosophy ('appreciate avant-garde art,' 'ponder existential questions'), and enthusiasm for complexity ('love solving puzzles'). Sentences often express curiosity ('I wonder why...') and tolerance for uncertainty. They may use poetic language and discuss emotions with depth ('feel joy in small things').",
  "partially high": "People partially high in Openness show cautious curiosity but retain some traditional preferences. They might express openness to 'some changes' or 'certain new ideas' while emphasizing comfort zones ('but within limits'). Their language blends practicality with occasional creativity ('innovation can work if proven'). They may discuss art or philosophy superficially ('appreciate classic movies') but avoid deeply abstract topics. Sentences use qualifiers ('maybe try new things once in a while') and mixed signals ('I like routine but sometimes crave novelty'). They acknowledge innovation but prioritize proven methods ('technology should improve life, but old tools still work').",
  "partially low": "People partially low in Openness prioritize familiarity and practicality. Their language favors tradition ('stick to what works,' 'old methods are reliable') and dismisses novelty ('why fix what isn’t broken?'). They avoid abstract discussions ('I prefer concrete plans'), focus on routine ('same schedule every day'), and resist emotional depth ('I don’t dwell on feelings'). Sentences emphasize predictability ('keep things simple') and skepticism toward unconventional ideas ('those ideas are unrealistic'). They may tolerate minor changes but reject major disruptions ('a new job might be too risky').",
  "low": "People low in Openness use rigid, conventional language rejecting novelty. They criticize change ('new ways are worse'), dismiss abstract thinking ('I don’t do philosophy'), and express discomfort with ambiguity ('there’s only one right answer'). Their communication avoids artistic or imaginative references, focusing on practicality ('facts over feelings'). Sentences are direct and concrete ('follow the rules') with frequent references to tradition ('this is how it’s always been'). They may distrust innovation ('technology causes more problems') and emotionally neutralize experiences ('I don’t need art to feel happy')."
}
]
```

-----

**Task**: According to the given texts, write a user post style summary that captures the personality types of the author. The summary should be comprehensive and concise.

**Requirements**:
- The result should be clear, concise and easy to understand, the summary also should indicate the given personality traits of the author.
- The result should follow the output format strictly without any errors explanations, or inconsistencies.

**Input Format**:
```markdown
Author's Personality: partially high Extroversion,  partially low Neuroticism, High Agreeableness, High Conscientiousness, High Openness.
---
Author's Texts: 
Post 1: [author's social post 1]
Post 2: [author's social post 2]
...
```

**Output Format**:
```plain
Language Style Summary: `your summary here`
```
-----

**Example for Clarify**:

Input Example:
```markdown
Author's Personality: partially high Extroversion,  partially high Neuroticism, partially high Agreeableness, partially high Conscientiousness, high Openness.
---
Author's Texts:
Post 1: It's time I fire up this life I'm livin' in!
Post 2: Cuando la vida misma te ha puesto en un aprieto, ¿qué es lo que se debe hacer?
Post 3: Que es lo que en realidad se debe de hacer? Lo que el corazón dicta o lo que la razon te dice?
Post 4: Es momento de aprender a dejar ir las cosas. Por favor, que aquella luz, guía mía, me lleve a los lugares indicados.
Post 5: Oscilando entre la infinidad de la mente y la infinidad del sueño...
Post 6: Hay veces en las que simplemente quisiera olvidarme de todo y refugiarme en la profundidad de mi mente...
Post 7: Meus somnerium... En busca de un pequeño momento de comprensión...
Post 8: De vuelta al mundo interno sin límites!
Post 9: Por favor, que las cosas mejoren! Si es que en verdad alguien puede ayudar, que le otorguen lo que necesita!
Post 10: Getting ready for the fun! Escapada espiritual al corazón del oráculo... Riviera Maya! Meus somnierum!
Post 11: I want to reconcile the violence in your heart! I want to recognise your beauty is not just a mask! I want to exorcise the demons from your past! I want to satisfy the undisclosed desires in your heart! (Gracias a Sofi por la canción tan genial!)
Post 12: I just... Don't know what to think anymore...
Post 13: Life truly has special ways to show us what we need... in the weirdest moments////f i n a l f a n t a s y////"I know. I'm not alone... not anymore."
Post 14: Back to the routine, then...
```

Output Example:
```plain
The author's language blends enthusiastic social engagement with situational restraint (partially high Extroversion), displaying emotional fluctuations tempered by cautious expressions (partially high Neuroticism). Cooperative empathy emerges but is balanced with self-focused ambivalence (partially high Agreeableness), alongside flexible organization and sporadic impulsivity (partially high Conscientiousness). Their style is intensely imaginative, embracing abstract metaphors, multilingual musings, and existential exploration (high Openness). The tone oscillates between vibrant urgency and introspective vulnerability, anchored in poetic curiosity and a quest for meaning.
```

"""

def judge_trait(score: float):
    if score < 0.25:
        return 'low'
    elif score < 0.5:
        return 'partially low'
    elif score < 0.75:
        return 'partially high'
    else:
        return 'high'


def language_style_analysis(idx, model, client, input_text):
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': input_text}]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.7)
    return idx, input_text, response


def construct_ref_facebook_library(data_path: str, model: str, base_url: str, api_key: str, output_path: Optional[str] = None):
    # load referenced Facebook data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # init medium results
    language_style_list = []
    input_texts = []

    # preprocess language style analysis process
    for i_texts, label in zip(data['text'], data['label_dists']):
        new_label = [judge_trait(i / 7.0) for i in label]
        language_style_prompt = "\n".join([trait_language_style_json[idx][score] for idx, score in enumerate(new_label)])
        language_style_list.append(language_style_prompt)
        input_string = f"Author's Personality: {new_label[0]} Extroversion, {new_label[1]} Neuroticism, {new_label[2]} Agreeableness, {new_label[3]} Conscientiousness, {new_label[4]} Openness\n---\n"
        input_string += "\n".join([f"Post {idx + 1}: {text}" for idx, text in enumerate(i_texts)]).strip()
        input_texts.append(input_string)

    # language style analysis process
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    response_list = [None for _ in range(len(input_texts))]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(language_style_analysis, idx, model, client, input_text) for idx, input_text in
                   enumerate(input_texts)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, input_text, response = future.result()
            response_list[idx] = response.choices[0].message.content.strip()
            if "```" in response_list[idx]:
                response_list[idx] = extract_code_from_plain(response_list[idx])
    data['language_style_knowledge'] = language_style_list
    data['language_style_analysis'] = response_list

    if output_path is not None:
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
    return data


def construct_ref_reddit_library(data_path: str, model: str, base_url: str, api_key: str, output_path: Optional[str] = None):
    # load referenced Facebook data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # init medium results
    language_style_list = []
    input_texts = []

    # preprocess language style analysis process
    for i_texts, label in zip(data['text'], data['label_dists']):
        new_label = [judge_trait(i) for i in label]
        language_style_prompt = "\n".join([trait_language_style_json[idx][score] for idx, score in enumerate(new_label)])
        language_style_list.append(language_style_prompt)
        input_string = f"Author's Personality: {new_label[0]} Extroversion, {new_label[1]} Neuroticism, {new_label[2]} Agreeableness, {new_label[3]} Conscientiousness, {new_label[4]} Openness\n---\n"
        input_string += "\n".join([f"Post {idx + 1}: {text}" for idx, text in enumerate(i_texts)]).strip()
        input_texts.append(input_string)

    # language style analysis process
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    response_list = [None for _ in range(len(input_texts))]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(language_style_analysis, idx, model, client, input_text) for idx, input_text in
                   enumerate(input_texts)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, input_text, response = future.result()
            response_list[idx] = response.choices[0].message.content.strip()
            if "```" in response_list[idx]:
                response_list[idx] = extract_code_from_plain(response_list[idx])
    data['language_style_knowledge'] = language_style_list
    data['language_style_analysis'] = response_list

    if output_path is not None:
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
    return data
