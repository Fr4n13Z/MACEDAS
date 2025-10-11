import re
from collections import OrderedDict
from transformers import PretrainedConfig
from transformers import XLMRobertaForMaskedLM, XLMRobertaForSequenceClassification

from .configuration_xlm_roberta import XLMRobertaFlashConfig as BertConfig
from .modeling_xlm_roberta import XLMRobertaForMaskedLM as FlashXLMRobertaForMaskedLM
from .modeling_xlm_roberta import XLMRobertaForSequenceClassification as FlashXLMRobertaForSequenceClassification
import torch

import click

## inspired by https://github.com/Dao-AILab/flash-attention/blob/85881f547fd1053a7b4a2c3faad6690cca969279/flash_attn/models/bert.py


def remap_state_dict(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a Huggingface BERT model to be flash_attn compatible.
    """

    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    state_dict = OrderedDict(
        (key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items()
    )

    # Layers
    def key_mapping_layers(key):
        return re.sub(r"^roberta.encoder.layer.", "roberta.encoder.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^roberta.embeddings.LayerNorm.", "roberta.emb_ln.", key)
        key = re.sub(
            r"^roberta.encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)",
            r"roberta.encoder.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^roberta.encoder.layers.(\d+).output.LayerNorm.(weight|bias)",
            r"roberta.encoder.layers.\1.norm2.\2",
            key,
        )
        key = re.sub(
            r"^cls.predictions.transform.LayerNorm.(weight|bias)",
            r"cls.predictions.transform.layer_norm.\1",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^roberta.encoder.layers.(\d+).intermediate.dense.(weight|bias)",
            r"roberta.encoder.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^roberta.encoder.layers.(\d+).output.dense.(weight|bias)",
            r"roberta.encoder.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    last_layer_subset = getattr(config, "last_layer_subset", False)
    for d in range(config.num_hidden_layers):
        Wq = state_dict.pop(f"roberta.encoder.layers.{d}.attention.self.query.weight")
        Wk = state_dict.pop(f"roberta.encoder.layers.{d}.attention.self.key.weight")
        Wv = state_dict.pop(f"roberta.encoder.layers.{d}.attention.self.value.weight")
        bq = state_dict.pop(f"roberta.encoder.layers.{d}.attention.self.query.bias")
        bk = state_dict.pop(f"roberta.encoder.layers.{d}.attention.self.key.bias")
        bv = state_dict.pop(f"roberta.encoder.layers.{d}.attention.self.value.bias")
        if not (last_layer_subset and d == config.num_hidden_layers - 1):
            state_dict[f"roberta.encoder.layers.{d}.mixer.Wqkv.weight"] = torch.cat(
                [Wq, Wk, Wv], dim=0
            )
            state_dict[f"roberta.encoder.layers.{d}.mixer.Wqkv.bias"] = torch.cat(
                [bq, bk, bv], dim=0
            )
        else:
            state_dict[f"roberta.encoder.layers.{d}.mixer.Wq.weight"] = Wq
            state_dict[f"roberta.encoder.layers.{d}.mixer.Wkv.weight"] = torch.cat(
                [Wk, Wv], dim=0
            )
            state_dict[f"roberta.encoder.layers.{d}.mixer.Wq.bias"] = bq
            state_dict[f"roberta.encoder.layers.{d}.mixer.Wkv.bias"] = torch.cat(
                [bk, bv], dim=0
            )

    def key_mapping_attn(key):
        return re.sub(
            r"^roberta.encoder.layers.(\d+).attention.output.dense.(weight|bias)",
            r"roberta.encoder.layers.\1.mixer.out_proj.\2",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r"^cls.predictions.bias", "cls.predictions.decoder.bias", key)

    state_dict = OrderedDict(
        (key_mapping_decoder_bias(k), v) for k, v in state_dict.items()
    )

    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["roberta.embeddings.word_embeddings.weight"]
        state_dict["roberta.embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
        )
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        state_dict["cls.predictions.decoder.weight"] = F.pad(
            decoder_weight, (0, 0, 0, config.vocab_size - decoder_weight.shape[0])
        )
        # If the vocab was padded, we want to set the decoder bias for those padded indices to be
        # strongly negative (i.e. the decoder shouldn't predict those indices).
        # TD [2022-05-09]: I don't think it affects the MLPerf training.
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        state_dict["cls.predictions.decoder.bias"] = F.pad(
            decoder_bias, (0, config.vocab_size - decoder_bias.shape[0]), value=-100.0
        )

    return state_dict


@click.command()
@click.option('--model_name', default='FacebookAI/xlm-roberta-base', help='model name')
@click.option('--revision', default='main', help='revision')
@click.option('--task', default='masked_lm', help='task')
@click.option('--output', default='converted_roberta_weights.bin', help='model name')
def main(model_name, revision, task, output):
    
    if task == 'masked_lm':
        roberta_model = XLMRobertaForMaskedLM.from_pretrained(model_name, revision=revision)
    elif task == 'sequence_classification':
        roberta_model = XLMRobertaForSequenceClassification.from_pretrained(model_name, revision=revision,num_labels=1)
    config = BertConfig.from_dict(roberta_model.config.to_dict())
    state_dict = roberta_model.state_dict()
    new_state_dict = remap_state_dict(state_dict, config)
    
    if task == 'masked_lm':
        flash_model = FlashXLMRobertaForMaskedLM(config)
    elif task == 'sequence_classification':
        flash_model = FlashXLMRobertaForSequenceClassification(config)

    for k, v in flash_model.state_dict().items():
        if k not in new_state_dict:
            print(f'Use old weights from {k}')
            new_state_dict[k] = v

    flash_model.load_state_dict(new_state_dict)

    torch.save(new_state_dict, output)


if __name__ == '__main__':
    main()
