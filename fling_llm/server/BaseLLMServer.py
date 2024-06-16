import os
from typing import Dict

from torch.utils.data import Dataset
from transformers import TrainingArguments
from fling.utils.registry_utils import SERVER_REGISTRY
from fling.component.server import BaseServer

from fling_llm.client.trainer import get_trainer


@SERVER_REGISTRY.register('base_llm_server')
class BaseLLMServer(BaseServer):

    def __init__(self, args: Dict, test_dataset: Dataset):
        super(BaseLLMServer, self).__init__(args, test_dataset)
        self.args = args
        self.test_dataset = test_dataset
        self.training_args = TrainingArguments(
            output_dir=os.path.join(args.other.logging_path, 'server'),
            per_device_train_batch_size=args.learn.batch_size,
            per_device_eval_batch_size=2 * args.learn.batch_size,
            **args.learn.hf_args
        )

    def test(self, model, test_dataset=None):
        model.eval()
        model.to(self.device)

        if test_dataset is None:
            test_dataset = self.test_dataset

        trainer = get_trainer(self.args.learn.trainer.name, model, train_dataset=None,
                              test_dataset=test_dataset, training_args=self.training_args)

        eval_res = trainer.evaluate(test_dataset)
        model.to('cpu')
        return eval_res
