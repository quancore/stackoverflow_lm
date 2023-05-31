This is a multilabel classification prototype on assigning correct labels to Stackoverflow questions using question title and body.

## Project info
- Date: 31/05/2023
- Done by: Baran Nama
- Requested By: Coresystems
- Dataset: https://www.kaggle.com/datasets/stackoverflow/stacksample?datasetId=265

## Folder structure
- **utils**: Python module for some small common utility functions among notebooks.
- **notebooks**: Jupyter notebooks which have been sorted by subtask order.
- **models**: Any trained models will be saved here.
- **datasets**: Input and processed datasets will be stored here.

## Notebook structure
- **01_data_processing**: Data cleaning operations has been made such as filtering by question score and question label frequency, basic visualizations and text cleaning.
- **02_baseline_model**: Several shallow classifiers such as logistic regression, tree based and gradient boosting based classifiers have been chained with TF-IDF vectorization.
- **03_lm_model**: A small Transformer based LM has been evaluated on zero shot setup as well as fine-tuned.
- **04_llm_model**: A LLM model has been fine-tuned using LoRA and State-of-the-art Parameter-Efficient Fine-Tuning (PEFT).

## Steps to reproduce the results and running notebooks
1. Download the dataset from the above URL, create a folder called ``datasets`` and unpack the downloaded dataset. There should be 3 CSV files.
2. Setup Conda environment using ``environment.yml`` file provided.
3. Register the environment to Jupyter and select the environment while running notebooks.
```bash
 python -m ipykernel install --user --name stackoverflow
```