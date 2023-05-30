import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.metrics import hamming_loss, f1_score, jaccard_score, roc_auc_score, accuracy_score

from transformers import EvalPrediction


# ## data pre-processing ###
def compare_body(df, before_col, after_col, n_samples=3):
    """ Compare two text columns before-after processing"""
    sample = df[[before_col, after_col]].sample(n=n_samples).values.tolist()
    for raw, cleaned in sample:
        print(f"*******Raw text: {before_col}*******\n{raw}\n*******Cleaned text:{after_col} *******\n{cleaned}\n##########\n")


# ## baseline model ###
def output_score(clf, y_pred, y_test):
    """ Return evaluation metrics as dataframe """
    performance = {"Jaccard": jaccard_score(y_test, y_pred, average="weighted"),
                   "Humming": hamming_loss(y_test, y_pred)*100,
                   "F1": f1_score(y_test, y_pred, average="weighted"),
                  }
    return pd.DataFrame(performance, index=[clf.__class__.__name__])


# ## LLM modelling ###
class StackOverflowDS(Dataset):
    """ Custom dataset implementation """
    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        cleaned_text = str(self.text[index])

        inputs = self.tokenizer.encode_plus(
            cleaned_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            # padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.float)
        }

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    """ Compute different metrics for language models"""
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    # return as dictionary
    metrics = {'F1': f1_score(y_true, y_pred, average="weighted"),
               'ROC_AUC': roc_auc_score(y_true, y_pred, average = 'weighted'),
               'Hamming': hamming_loss(y_true, y_pred)*100,
               'Jaccard': jaccard_score(y_true, y_pred, average="weighted"),
               'Accuracy': accuracy_score(y_true, y_pred)}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result