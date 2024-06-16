from easydict import EasyDict

default_exp_args = dict(
    data=dict(
        dataset='shakespear', data_path='Trelis/tiny-shakespeare',
        tokenizer='openai-community/gpt2',
        max_len=512,
        sample_method=dict(name='iid', train_num=100, test_num=20)
    ),
    learn=dict(
        device='cuda:0', local_eps=5, global_eps=40, batch_size=32,
        trainer=dict(name='sft_fedavg_trainer'),
        # Learning rate scheduler. For each global epoch, use a dynamic learning rate.
        scheduler=dict(
            # Default to be "fix", which means learning rate used in each global epoch is identical.
            name='fix'
        ),
        hf_args=dict(
            group_by_length=False,
            dataloader_pin_memory=False,
            evaluation_strategy="no",
            save_strategy="no",
            report_to='none',
            remove_unused_columns=False,
        )
    ),
    model=dict(
        model_path='openai-community/gpt2',
    ),
    client=dict(name='base_llm_client', client_num=10, sample_rate=1, val_frac=0),
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
    other=dict(test_freq=1, logging_path='./logging/shakespear_fedavg_gpt2', print_config=False, resume_path=None)
)

default_exp_args = EasyDict(default_exp_args)
