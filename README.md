# 🧬 SwarmPrompt

This is a PSO-based modification of the EvoPrompt framework introduced in the paper [Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers](https://arxiv.org/abs/2309.08532). 

## 📃 Abstract

Prompt engineering is crucial for optimizing large language models (LLMs), yet manual design is challenging due to high-dimensional prompt spaces. We introduce \textbf{SwarmPrompt}, a framework that integrates Particle Swarm Optimization (PSO) into EvoPrompt for automated prompt optimization. Unlike traditional PSO, SwarmPrompt operates in a \textit{semantic search space}, where prompts evolve via structured social exchange. Crucially, it leverages DeepSeek-R1-Distill-Qwen-32B—optimized for instruction-following—to refine prompts more effectively than GPT-4. On ASSET, a prominent dataset of text simplification, SwarmPrompt achieves a 55.53 SARI score, surpassing EvoPrompt’s Differential Evolution baseline by 17.1\%. Our findings portray prompt engineering as a \textit{continuous} optimization problem, where small refinements drive significant gains, offering a scalable solution for automating instruction tuning in LLMs. 
## 🚀 Quick Start

### ⚙️ Preparation

1. **Environmental** settings: `pip install -r requirements.txt`
2. **Data** download: The test data for the language understanding task can be found [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Put the test file in the folder `./data/cls/{dataset_name}`. For datasets of BBH, download from the repo [CoT-hub](https://github.com/FranxYao/chain-of-thought-hub/tree/main/BBH/data) and put them in the folder `BBH/data/{dataset_name}`.
3. **OpenAI API key** required: add your OpenAI API key and other related settings in the file `auth.yaml`

### ♻ Evolution

We instanciate two evolutionary algorithms, GA (genetic algorithm) and DE (diffenrential evolution) to evolve upon the initial population. Evolve your prompts using the following commands:

Customize the parameter `--llm_type` to use `text-davinci-003`, `gpt-3.5-turbo`, `gpt-4`.

```bash

# simplification task on GPT-4
bash scripts/sim/run_de_gpt.sh
bash scripts/sim/run_ga_gpt.sh
bash scripts/sim/run_pso_gpt.sh
```



### 📌 Notes

Note that we have two language models used in our framework, one is for evolution (argument `--llm_type`), the other for the task implementation (`--language_model`).

#### 💡Tips for Usage

The number of iteration and the population size effect the performance of EvoPrompt/SwampPrompt. There exists a trade-off between the cost and the performance. For relative simple tasks, a size of 10 and 10 iterative steps are enough, or even less. While for complex tasks, a larger population with diversity is required.

#### 🔢 Arguments

You may need to set the following arguments to customize your own configuration.

- `task`: the task category, such as `sim` (simplification), `cls`(classification), `sum`(summarization). If you need to extend this to other tasks, you may override the metric to evaluate
- `dataset`: the dataset you want to evolve prompt on
- `dev_file`: the path of the devlopment set
- `language model`: the model used for task implementation
- `llm_type`: the LLM used to evolve prompts
- `position`: this argument mainly indicates whether to use demonstration (zero-shot or few-shot)
- `sample_num`: the size of dev set, mainly used for generation task where there is no need to set the `dev_file`
- `prompt_num`: number of examples for few-shot demonstrations

## 🌳 Code Strucutre

```python
.
├── args.py
├── auth.yaml
├── BBH  # code for BBH tasks
├── data  # dataset, templates used
│   ├── cls
│   ├── sim
│   ├── sum
│   ├── template_de.py  # templates of prompt evolution by DE
│   ├── template_ga.py  # templates of prompt evolution by GA
│   ├── template_v2.json  # templates for task implementation
│   └── templates.py  # wrapper
├── dataset.py  # dataset class
├── evaluator.py  # evaluators on different tasks
├── evoluter.py  # DE, GA, APE, PSO
├── evolution.py  # DE, GA, APE, PSO 
├── get_result.py
├── infer.py  # main file for inference
├── llm_client.py  # LLM query
├── metrics.py  # metric calculation
├── requirements.txt
├── run.py  # main file for evolution
├── scripts  # scripts to run the code
└── utils.py  # auxiliary functions
```

## 🧩 Possible Extension

- **Aggregation**: Based on the final population of high quality, ensembling strategies can be effectively applied upon the prompts.
- **More fine-grained metrics**: to select prompt maintained in the population, we need to evaluate the performance on dev set. However, for understanding tasks, metrics such as accuracy or F1 are coarse-grained, sometimes it's not accurate anough to select which to keep in the population since the performances of them are the same.
- **More complex tasks** are left to explore.


## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@article{guo2023connecting,
  title={Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers},
  author={Guo, Qingyan and Wang, Rui and Guo, Junliang and Li, Bei and Song, Kaitao and Tan, Xu and Liu, Guoqing and Bian, Jiang and Yang, Yujiu},
  journal={arXiv preprint arXiv:2309.08532},
  year={2023}
}
```

## Acknowledgements

Our codebase is based on the following repos. Thanks for open-sourcing!

- [CoT-hub](https://github.com/FranxYao/chain-of-thought-hub)
- [APE](https://github.com/keirp/automatic_prompt_engineer)
- [LM-BFF](https://github.com/princeton-nlp/LM-BFF)

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
