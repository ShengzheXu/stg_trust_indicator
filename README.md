# Is Isotropy a Good Proxy for Generalization in Time Series Forecasting with Transformers?

## 1. Preparation

### Example Datasets for Quantitative and Qualitative Analysis


| Dataset | Domain | Freq. | Num. Series | Min Length | Avg Length | Max Length | Prediction Length (H) | Link |
|--------|--------|-------|-------------|-------------|-------------|-------------|------------------------|------|
| Australian Electricity | Energy | 30min | 5 | 230,736 | 231,052 | 232,272 | 48 | [link](https://zenodo.org/record/4659727) |
| Car Parts | Retail | 1M | 2,674 | 51 | 51 | 51 | 12 | [link](https://zenodo.org/record/4656022) |
| Covid Deaths | Healthcare | 1D | 266 | 212 | 212 | 212 | 30 | [link](https://zenodo.org/record/4656009) |
| Dominick | Retail | 1D | 100,014 | 201 | 296 | 399 | 8 | [link](https://www.chicagobooth.edu/research/kilts/research-data/dominicks) |
| Exchange Rate | Finance | 1B | 8 | 7,588 | 7,588 | 7,588 | 30 | [link](https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate) |
| FRED-MD | Economics | 1M | 107 | 728 | 728 | 728 | 12 | [link](https://zenodo.org/records/4654833) |
| Hospital | Healthcare | 1M | 767 | 84 | 84 | 84 | 12 | [link](https://zenodo.org/record/4656014) |
| NN5 (Weekly) | Finance | 1W | 111 | 113 | 113 | 113 | 8 | [link](https://zenodo.org/records/4656125) |
| Weather | Nature | 1D | 3,010 | 1,332 | 14,296 | 65,981 | 30 | [link](https://zenodo.org/record/4654822) |
| Transportation Signal | Transport | 1D | 3,010 | 1,332 | 14,296 | 65,981 | 30 | [link](https://zenodo.org/record/4654822) |
| Synthetic (10 kernels) | Numerical | - | 1,000,000 | 1,024 | 1,024 | 1,024 | 64 | [link](https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/kernel-synth.py) |

### Model Installation

Due to significant differences in model dependencies, we highly recommend using a package manager (e.g., `conda` or `venv`) and setting up separate Python environments for each tested model. This ensures compatibility and prevents conflicts during installation and execution.

Note: This recommendation applies only to running third-party object models. Our method, `stg_trust_indicator`, is lightweight and does not require complex dependencies.


The installation instructions for each model can be found in their official repositories:

- **Chronos-T5** and **Chronos-Bolt**  
  [https://github.com/amazon-science/chronos-forecasting/tree/main](https://github.com/amazon-science/chronos-forecasting/tree/main)

- **PatchTST**  
  [https://github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)

- **Moirai**  
  [https://github.com/redoules/moirai](https://github.com/redoules/moirai)

- **Lag_LLaMA**  
  [https://github.com/time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama)

- **LLM4CP (LLM for Channel Prediction)**  
  [https://github.com/liuboxun/LLM4CP/tree/master](https://github.com/liuboxun/LLM4CP/tree/master)

## 2. Computing and Visualizing Embedding Isotropy

- **Chronos-T5** and **Chronos-Bolt** 

```sh
source activate isotropy

# chronos-t5
python stg_trust_indicator.py --num_experiments 10 --input_len 500 --output_len 64 --num_rows 400 --model_name amazon/chronos-t5-small --data_file ../data/single_kernelsynth_1000.arrow --result_all_folder result_folder
python stg_trust_indicator.py --num_experiments 10 --model_name amazon/chronos-t5-small --real_data standard --num_rows 100 --result_all_folder result_folder

# chronos-bolt
python stg_trust_indicator.py --num_experiments 10 --input_len 500 --output_len 64 --num_rows 400 --model_name amazon/chronos-bolt-small --data_file ../data/single_kernelsynth_1000.arrow --result_all_folder result_folder
python stg_trust_indicator.py --num_experiments 10 --model_name amazon/chronos-bolt-small --real_data standard --num_rows 100 --result_all_folder result_folder

# patchtst
python stg_trust_indicator.py --num_experiments 10 --input_len 500 --output_len 64 --num_rows 2000 --model_name patchtst --data_file ../data/single_kernelsynth_1000.arrow --result_all_folder result_folder
python stg_trust_indicator.py --num_experiments 10 --model_name patchtst --real_data standard --num_rows 2000 --result_all_folder result_folder
```
- **Moirai** and **Lag_llama**  
These models typically require a different environment due to dependency differences.

```sh
conda deactivate
source activate moirai
# moirai
python stg_trust_indicator.py --num_experiments 10 --input_len 500 --output_len 64 --num_rows 1000 --model_name moirai --data_file ../data/single_kernelsynth_1000.arrow --result_all_folder result_folder
python stg_trust_indicator.py --num_experiments 10 --model_name moirai --real_data standard --num_rows 1000 --result_all_folder result_folder

conda deactivate
source activate lag_llama
# lag_llama
python stg_trust_indicator.py --num_experiments 10 --input_len 500 --output_len 64 --num_rows 400 --model_name lag_llama --data_file ../data/single_kernelsynth_1000.arrow --result_all_folder result_folder
python stg_trust_indicator.py --num_experiments 10 --model_name lag_llama --real_data standard --num_rows 700 --result_all_folder result_folder
```

- **LLM4CP**  
Generative model for Channel Prediction.

```sh
# llm4cp
python stg_trust_indicator.py --num_experiments 10 --model_name llm4cp_fdd --num_rows 500
python stg_trust_indicator.py --num_experiments 10 --model_name llm4cp_tdd --num_rows 500
```

# 3. Citation

If you found our work useful, please cite our work.

```
@article{
shelim2025is,
title={Is isotropy a good proxy for generalization in time series forecasting with transformers?},
author={Rashed Shelim and Shengzhe Xu and Walid Saad and Naren Ramakrishnan},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=iUtDYVQzFq},
note={}
}
```
