import copy
import easydict

import torch.nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def export_hf_model(model_args: easydict.EasyDict) -> torch.nn.Module:
    """
    Overview:
        Given the model arguments, initialize the model using the api of ``transformers``.
    Arguments:
        - model_args: The given model arguments.
    Returns:
        - model: The constructed model.
    """
    # Extract some arguments.
    tmp_args = copy.deepcopy(model_args)
    path = tmp_args.pop('model_path')
    pretrained = tmp_args.pop('pretrained')

    # If ``pretrained`` is true, use the parameter in ``path`` to initialize the model.
    # Otherwise, the model will be initialized randomly with the same architecture as the model in ``path``.
    if pretrained:
        return AutoModelForCausalLM.from_pretrained(path, **tmp_args)
    else:
        config = AutoConfig.from_pretrained(path)
        return AutoModelForCausalLM.from_config(config, **tmp_args)


def export_hf_tokenizer(tokenizer_path, **kwargs) -> transformers.PreTrainedTokenizer:
    """
    Overview:
        Given the tokenizer arguments, initialize the tokenizer using the api of ``transformers``.
    Arguments:
        - tokenizer_path: The path of tokenizer. This can be either a local path or a huggingface url.
    Returns:
        - tokenizer: The constructed pretrained tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
