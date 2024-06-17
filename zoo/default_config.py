from easydict import EasyDict

default_exp_args = dict(
    # Configurations for dataset.
    data=dict(
        # Name for dataset. Must be registered in `DATASET_REGISTRY`.
        dataset='shakespear',
        # The path for loading data. This value can be either a local path or huggingface url.
        data_path='Trelis/tiny-shakespeare',
        # The tokenizer used for tokenize dataset. This value can be either a local path or huggingface url.
        tokenizer='openai-community/gpt2',
        # Max length for tokenizing text.
        max_len=512,
        # How datasets distribute across all clients.
        sample_method=dict(
            name='iid',
            # Default training number for each client is 100.
            train_num=100,
            # Default testing number for each client is 20.
            test_num=20
        )
    ),
    learn=dict(
        # Running device for deep learning model. If only CPU is available, set this key to be "cpu".
        device='cuda:0',
        # Number of local epochs in each training round of each client.
        local_eps=5,
        # Number of global epochs (training rounds) in the total FL process.
        global_eps=40,
        # Batch size for local training. Testing batch size is 2 times as large as this value
        batch_size=32,
        # Trainer used, which should be found in ``fling_llm.client.trainer``.
        trainer=dict(name='sft_fedavg_trainer'),
        # Test place for federated learning. Options: 'before_aggregation', 'after_aggregation'
        test_place=['after_aggregation'],
        # Learning rate scheduler. For each global epoch, use a dynamic learning rate.
        scheduler=dict(
            # Default to be "fix", which means learning rate used in each global epoch is identical.
            name='fix',
            # Base learning rate for scheduling lr.
            base_lr=5e-5
        ),
        # Huggingface arguments when initializing trainer.
        # Detailed descriptions for these values can be found at:
        # https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        hf_args=dict(
            group_by_length=False,
            dataloader_pin_memory=False,
            evaluation_strategy="no",
            save_strategy="no",
            report_to='none',
            remove_unused_columns=False,
            disable_tqdm=True
        )
    ),
    model=dict(
        # Path for loading model. This value can be either a local path or huggingface url.
        model_path='openai-community/gpt2',
        # Whether to load pretrained parameters.
        pretrained=True,
    ),
    # Configurations for client.
    client=dict(
        # Name for client used. Must be registered in `CLIENT_REGISTRY`.
        name='base_llm_client',
        # Number of clients.
        client_num=10,
        # The ratio of clients participated in each global epoch. For instance, if `sample_rate=0.5`,
        # only half of all clients will join federated learning in each global epoch.
        sample_rate=1,
        # The fraction ratio of test samples in total samples. For instance, if `val_frac=0.2`, this means
        # 20% of total data samples will be regarded as local validation dataset, and 80% for training dataset.
        val_frac=0
    ),
    # Configurations for server.
    server=dict(name='base_llm_server'),
    group=dict(
        # Name for group used. Must be registered in `GROUP_REGISTRY`.
        name='base_group',
        # How parameters in each client aggregate. Default to be "avg", which means a simple average.
        aggregation_method='avg',
        # What parameters in each client should be aggregated.
        aggregation_parameters=dict(
            # For default case, every parameter should be aggregated.
            name='all'
        ),
    ),
    # Launcher configurations.
    launcher=dict(
        # Currently, we only support 'serial' launcher.
        name='serial'
    ),
    # Other configurations.
    other=dict(
        # Frequency for testing. For example, `test_freq=3` means the performance is tested every 3 global epochs.
        test_freq=1,
        # The logging directory of this experiment.
        # If the directory does not exist, it will be created automatically.
        # If the directory already exists, some parts might be over-written, which should be carefully inspected.
        logging_path='./logging/shakespear_fedavg_gpt2',
        # Whether to print config is the command line.
        print_config=False,
        # The saved model checkpoint to start from. If it is set to ``None``, the training process
        # will start from scratch.
        resume_path=None
    )
)

default_exp_args = EasyDict(default_exp_args)
