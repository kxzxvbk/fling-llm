## Fling-LLM

**Fling-LLM** is a research platform for Federated Learning on Large Language Models (LLMs).

Its goal is to enable researchers to run LLMs in federated learning, allowing for quick, accurate, and convenient training and testing of federated learning algorithms + LLMs on common language model data. For the large language model part, we use PyTorch and transformers as the backend for training; for the federated learning part, we use Fling as the backend for training.

It mainly supports:

- Various Federated Learning methods, such as FedAvg, FedProx, ...
- Various LLM tasks, such as SFT, instruct tuning, ...

## Installation

Firstly, it is recommended to install PyTorch manually with a suitable version (we recommend using PyTorch>=2.0). Instructions for installation can be found at this [link](https://pytorch.org/get-started/locally/).

After the first step, you can install the latest version of Fling with the following command by using Git:

```bash
git clone https://github.com/kxzxvbk/Fling
cd Fling
pip install -e .
```

You can use ``fling -v`` to check whether Fling is successfully installed.

Finally, you need to install the code from this repository:

```shell
git clone https://github.com/kxzxvbk/Fling-llm
cd Fling-llm
pip install -e .
```

## Quick Start

After successfully installing Fling-llm, users can start the first experiment by using the following command. An example for generic federated learning:

```bash
python zoo/shakespear/shakespear_causal_lm_fedavg_config.py
```

This config is a simplified version for conducting FedAvg on the Shakespeare dataset for SFT training, and iterates for 4 communication rounds.

For other algorithms and datasets, users can refer to `zoo/` or customize their own configuration files.

## Feature

- Support for a variety of algorithms and datasets.
- Efficient execution, supporting multi-GPU parallel training.

## Supported Algorithms

### Supported Federated Learning Algorithms

**FedAvg:** [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

### Supported Language Model Tasks

**Supervised Fine-Tuning (SFT):** [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

## Feedback and Contribution

- For any bugs, questions, or feature requests, feel free to propose them in [issues](https://github.com/kxzxvbk/Fling-llm/issues).
- For any contributions that can improve Fling-llm (more algorithms or better system design), we warmly welcome you to propose them in a [pull request](https://github.com/kxzxvbk/Fling-llm/pulls).

## Acknowledgments

Special thanks to [@chuchugloria](https://github.com/chuchugloria), [@Kye](https://github.com/KyeGuo), [@XinghaoWu](https://github.com/XinghaoWu).

## Citation
```latex
@misc{Fling-llm,
    title={Fling-llm: Fling extension for Large Language Models (LLMs)},
    author={Fling-llm Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/kxzxvbk/Fling-llm}},
    year={2024},
}
```

## License
Fling-llm is released under the Apache 2.0 license.
