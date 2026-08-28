# TSSI

This repository provides the implementation of TSSI: **Temporal-Spectral Selective Interaction for Multi-Rate Multivariate Time Series Forecasting**.

## Getting Started

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

The five datasets used in our experiments can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) link provided by [Autoformer](https://github.com/thuml/Autoformer). Arrange the downloaded files as follows:

```text
datasets/
├── ETT-small/
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── electricity/
│   └── electricity.csv
└── traffic/
    └── traffic.csv
```

Set the project and dataset paths, then generate the multi-rate data:

```bash
export PROJECT_DIR="$(pwd)"
export DATASETS_DIR="/path/to/datasets"
bash scripts/data_resample.sh
```

To construct the multi-rate series, we apply downsampling operators according to the physical nature of each dataset: instantaneous interval sampling for ETTm1, ETTm2, and Weather, and cumulative block summation for Electricity and Traffic. The `vars_group.txt` files included under `resample_data/` define the variable-to-rate assignments used to produce the results reported in the paper. The preprocessing script generates the corresponding data splits at `resample_data/<dataset>/<ratio-config>/<resample-run>/<ratio>/{train,vali,test}.csv`.

## Running Experiments

We provide ready-to-use launch scripts for TSSI and the included baselines. Those scripts iterate over the configured datasets, sampling-ratio settings, resampling runs, and prediction lengths. They write logs, checkpoints, and prediction results to `logs/`, `checkpoints/`, and `results/`, respectively.

For example:

```bash
bash scripts/tssi.sh
bash scripts/itransformer.sh
```

## Acknowledgement

We thank the authors of the following repositories for making their code and datasets publicly available:

- [Are Transformers Effective for Time Series Forecasting?](https://github.com/cure-lab/LTSF-Linear)
- [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://github.com/thuml/iTransformer)
- [SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters](https://github.com/lss-1138/SparseTSF)
- [TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://github.com/thuml/TimeXer)
- [LETS Forecast: Learning Embedology for Time Series Forecasting](https://github.com/abrarmajeedi/DeepEDM) (DeepEDM)
- [CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables](https://github.com/mumiao2000/CrossLinear)
- [DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting](https://github.com/decisionintelligence/DUET)
- [Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework for Dependency, Asynchrony, and Missingness](https://github.com/jinkwan1115/ChannelTokenFormer) (ChannelTokenFormer)
- [MF-CLR: Multi-Frequency Contrastive Learning Representation for Time Series](https://github.com/duanjufang/MF-CLR)
- [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://github.com/Time-MoE/Time-MoE)
- [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://github.com/thuml/Autoformer)
