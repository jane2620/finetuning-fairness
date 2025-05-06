# Can Fine-tuning Aligned Models Degrade Fairness?

Contributors: Jane Castleman & Tanvi Namjoshi

This code is part of a project for [COS 598A: AI Safety & Alignment](https://princeton-polaris-lab.github.io/ai-safety-course/index.html) @ Princeton University. The project explores when fine-tuning on data can potentially lead to bias amplification in hiring decisions (see the Abstract below for more details).

## File Organization

`datasets/ft/` contains different datasets used to fine-tune the Llama models

`eval_datasets/` contains the code to extend the hiring scenarios from [Salinas et al.](https://arxiv.org/abs/2402.14875) for additional occupations and demographic groups. `hiring_prompts.csv` has the full list of evaluation prompts. 

`examples_for_later` can be ignored, it contains code that may be useful for future experiments

`finetuned_models/` contains a subset of the fine-tuned Llama models we used 

`get_salinas_results/` contains the scripts to evaluate our models 

`results/` contains the model outputs as well as analysis of our results. The most important python notebooks are `results/plotting_amplification`, `results/stat_analysis`, `results/plotting_no_bias_amp.ipynb`, and `results/new_metrics_testing.ipynb`. **To use these notebooks, first run:**
```
python results/get_all_seed_results.py --seed all --model Llama-3.1-8B-Instruct

python results/get_all_seed_results_no_bias.py --seed all --model Llama-3.1-8B-Instruct
```

## Requirements

```
pip install -r requirements.txt
```

To run the slurm jobs, you need a conda environment called `FairTune`


## Abstract

Fine-tuning is used extensively by model developers and downstream users to improve performance, customize general-purpose models for specific tasks, and improve fairness of model outputs given different inputs. However, recent work has shown the dangers of fine-tuning in compromising safety guardrails, both intentionally and unintentionally. 
We ask whether fine-tuning can compromise \textit{fairness alignment}, exacerbating model biases against protected attributes and amplifying unequal decisions over baseline models.
To study this phenomenon in detail, we look at decision-making in employment, prompting model Llama 3.1-8B-Instruct to estimate salaries for applicants with varying races and genders. Then, we compare baseline model unfairness with models fine-tuned on various unrelated and task-relevant datasets using low-rank adaptation (LoRA).
We find that fine-tuning can cause models to become significantly more biased in their salary estimations, particularly against Black and Hispanic women, when fine-tuned on both unrelated and task-relevant datasets. 
These results have important implications for downstream users of LLMs for employment-related decisions, as fine-tuning can unintentionally result in unfair models. Furthermore, emerging regulations surrounding bias audits for high-impact automated decision-making models should consider the potential for fine-tuning to degrade fairness.

