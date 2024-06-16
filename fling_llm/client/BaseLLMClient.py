import copy
import random

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from fling.utils import get_optimizer, VariableMonitor, get_weights
from fling.utils.registry_utils import CLIENT_REGISTRY

from fling_llm.client.trainer import get_trainer


@CLIENT_REGISTRY.register('base_llm_client')
class BaseLLMClient:
    """
    Overview:
    This class is the base implementation of client in Federated Learning.
    Typically, a client need to have these functions.
    ``train``: A client need to define the local training process.
    ``test``: A client need to define how to test the local model given a dataset.
    ``finetune``: A client need to define how to finetune the local model (usually used in Personalized FL)
    If users want to define a new client class, it is recommended to inherit this class.
    """

    def __init__(self, args: dict, model, client_id: int, train_dataset: Dataset, test_dataset: Dataset = None):
        """
        Overview:
            Initializing train dataset, test dataset(for personalized settings).
        Arguments:
            - args: dict type arguments.
            - train_dataset: private dataset for training
            - test_dataset: private dataset for testing (Optional)
            - client_id: unique id for this client.
        Returns:
            - None
        """
        # Model construction.
        self.args = args
        self.model = model
        self.device = args.learn.device
        # Specify a unique client id.
        self.client_id = client_id
        # This attribute will not be set until ``self.set_fed_keys(self, keys)`` is called.
        # Only weights in ``self.fed_keys`` will be collaboratively trained using Federated Learning.
        self.fed_keys = []
        val_frac = args.client.val_frac
        # If val_frac > 0, it means that a fraction of the given dataset will be separated for validating.
        if val_frac == 0:
            # ``self.sample_num`` refers to the number of local training number.
            self.sample_num = len(train_dataset)
            self.train_dataset = train_dataset
        else:
            # Separate a fraction of ``train_dataset`` for validating.
            real_train = copy.deepcopy(train_dataset)
            real_test = copy.deepcopy(train_dataset)
            # Get the indexes of train dataset.
            indexes = real_train.indexes
            random.shuffle(indexes)
            # Randomly sampling a part to be test dataset.
            train_index = indexes[:int((1 - val_frac) * len(train_dataset))]
            test_index = indexes[int((1 - val_frac) * len(train_dataset)):]
            real_train.indexes = train_index
            real_test.indexes = test_index
            # ``self.sample_num`` refers to the number of local training number.
            self.sample_num = len(real_train)

            self.train_dataset = real_train
            self.val_dataset = real_test

        if test_dataset is not None:
            self.test_dataset = test_dataset

        self.training_args = TrainingArguments(
            output_dir=os.path.join(args.others.logging_path, 'server'),
            evaluation_strategy="no",
            save_strategy="no",
            report_to='none',
            remove_unused_columns=False,
            per_device_train_batch_size=args.learn.batch_size,
            per_device_eval_batch_size=2 * args.learn.batch_size,
            group_by_length=False,
            dataloader_pin_memory=False,
        )

        self.trainer = get_trainer(self.args.learn.trainer.name, model, train_dataset=None,
                              test_dataset=test_dataset, training_args=self.training_args)

    def set_fed_keys(self, keys: Iterable) -> None:
        r"""
        Overview:
            Set `self.fed_dict` to determine which parameters should be aggregated.
        Arguments:
            - keys: sequence that contains the keys of parameters that need to be aggregated.
        Returns:
            - None
        """
        self.fed_keys = list(keys)

    def update_model(self, dic: dict) -> None:
        r"""
        Overview:
            Update the state_dict of the local model of this client.
            For keys not existed in the argument `dic`, the value will be retained.
        Arguments:
            - dic: dict type parameters for updating local model.
        Returns:
            - None
        """
        dic = copy.deepcopy(dic)
        state_dict = self.model.state_dict()
        state_dict.update(dic)

        self.model.load_state_dict(state_dict)

    def get_state_dict(self, keys: Iterable) -> dict:
        r"""
        Overview:
            Get the parameter diction of local model.
        Arguments:
            - keys: sequence that contains the keys of parameters that are acquired.
        Returns:
            - partial_dict: the acquired diction of parameters.
        """
        state_dict = self.model.state_dict()
        partial_dict = {k: state_dict[k] for k in keys}
        return partial_dict

    def train(self, lr, device=None, train_args=None):
        """
        Local training.
        """
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.train()
        self.model.to(self.device)

        res = self.trainer.train()
        metrics = res.metrics

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        if device is not None:
            self.device = device_bak

        return metrics

    def finetune(self, lr, finetune_args, device=None, finetune_eps=None, override=False):
        raise NotImplementedError()

    def test(self):
        """
        Test model.
        """
        self.model.eval()
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor()

        res = self.trainer.evaluate()

        # Put the model to cpu after training to save GPU memory.
        self.model.to('cpu')

        return res