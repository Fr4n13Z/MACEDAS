

## Environment

Python requirements:

```python
openai
emoji
pandas
scikit-learn
sentence_transformers
pytorch
vllm
```

Pretrained Models: `bert-base-uncased`, `jina-embedding-v3`, `RoBERTa-base-uncased` from the [HuggingFace](https://huggingface.co), and [glove.42B.300d](https://nlp.stanford.edu/projects/glove/). The pretrained model parameter files are put in ==./PTM== directory.

Dify Agent Configurations and the relied knowledge bases are in ==./Dify_apps== directory.

## Personality Alignment

run the personality alignment script:

```bash
python ./personality_align.py
```

## Data Synthesis

run the data synthesis script:

```bash
python ./text_synthesis.py
```

## Personality Detection Baseline Training

run the scripts in the ==./Persoanlity_Detection_Models==.

1. AttRCNN: 

   ```bash
   python ./AttRCNN_main.py
   ```

2. BERT+MLP:

   ```bash
   python ./BERT_main.py
   ```

3. D-DGCN:

   ```bash
   python ./DDGCN_main.py
   ```

4. DEN:

   ```bash
   python ./DEN_main.py
   ```

5. EERPD:

   ```bash
   python ./EERPD_main.py
   ```

6. MvP:

   ```bash
   python ./MvP_main.py
   ```

## Ethical Statements

Text-based personality detection is an ethically sensitive task aimed at the people themselves. Our study aims to develop a data-central method for personality analysis instead of creating a privacy-invading tool. To ensure the privacy of users, we worked within the purview of acceptable privacy practices and strictly followed the Facebook and Reddit data usage policies. The datasets of reference libraries used in our research are from public sources, and all user information is anonymized. Moreover, any research or application based on our study is only allowed for research purposes to avoid the misuse and abuse of our open-source code and data for causing real-world threats.
