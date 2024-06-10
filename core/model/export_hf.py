import copy
import easydict

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def export_hf_model(model_args: easydict.EasyDict, pretrained: bool):
    tmp_args = copy.deepcopy(model_args)
    path = tmp_args.pop('model_path')

    if pretrained:
        return AutoModelForCausalLM.from_pretrained(path, **tmp_args)
    else:
        config = AutoConfig.from_pretrained(path)
        return AutoModelForCausalLM.from_config(config, **tmp_args)


def export_hf_tokenizer(tokenizer_path, **kwargs):
    return AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
