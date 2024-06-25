from easydict import EasyDict

exp_args = dict(
    data=dict(
        dataset='shakespear',
        data_path='Trelis/tiny-shakespeare',
        tokenizer='openai-community/gpt2',
        max_len=512,
        sample_method=dict(name='uniform', train_num=0, test_num=0)
    ),
    learn=dict(
        device='cuda:0',
        local_eps=5,
        global_eps=40,
        batch_size=32,
        trainer=dict(name='sft_fedavg_trainer'),
    ),
    model=dict(
        model_path='openai-community/gpt2',
        pretrained=True,
    ),
    client=dict(name='base_llm_client', client_num=10),
    server=dict(name='base_llm_server'),
    group=dict(name='base_group', aggregation_method='avg'),
    other=dict(test_freq=1, logging_path='./logging/shakespear_fedavg_gpt2')
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling_llm.pipeline import generic_model_pipeline
    generic_model_pipeline(exp_args, seed=0)
