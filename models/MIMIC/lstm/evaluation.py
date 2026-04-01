import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

def bootstrap_test_evaluation(
    model,
    test_data_loader,
    loss_function,
    device="cuda",
    n_bootstrap=1000,
    csv_path="bootstrap_test_results.csv",
    seed=42,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    # ---- Get full test predictions once ----
    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            logits = model(input_ids, lengths)
            loss = loss_function(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    N = len(all_labels)

    # ---- Bootstrap ----
    bootstrap_results = []

    for i in range(n_bootstrap):

        indices = np.random.choice(N, N, replace=True)

        y_true = all_labels[indices]
        y_pred = all_preds[indices]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Short stay = class 0
        short_precision = precision_score(y_true, y_pred, pos_label=0)
        short_recall = recall_score(y_true, y_pred, pos_label=0)  # Sensitivity
        short_f1 = f1_score(y_true, y_pred, pos_label=0)

        short_specificity = tp / (tp + fn)  # recall of class 1

        # Long stay = class 1
        long_precision = precision_score(y_true, y_pred, pos_label=1)
        long_recall = recall_score(y_true, y_pred, pos_label=1)
        long_f1 = f1_score(y_true, y_pred, pos_label=1)

        long_specificity = tn / (tn + fp)

        bootstrap_results.append({
            "accuracy": accuracy_score(y_true, y_pred),

            "short_precision": short_precision,
            "short_recall": short_recall,
            "short_specificity": short_specificity,
            "short_f1": short_f1,

            "long_precision": long_precision,
            "long_recall": long_recall,
            "long_specificity": long_specificity,
            "long_f1": long_f1,

            "weighted_precision": precision_score(y_true, y_pred, average="weighted"),
            "weighted_recall": recall_score(y_true, y_pred, average="weighted"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        })

    df = pd.DataFrame(bootstrap_results)

    # ---- Summary (mean, std, CI) ----
    summary_rows = []

    for col in df.columns:
        values = df[col].values
        summary_rows.append({
            "metric": col,
            "mean": np.mean(values),
            "std": np.std(values),
            "ci_lower_95": np.percentile(values, 2.5),
            "ci_upper_95": np.percentile(values, 97.5),
        })

    summary_df = pd.DataFrame(summary_rows)

    df.to_csv("bootstrap_raw_results.csv", index=False)
    summary_df.to_csv(csv_path, index=False)

    print(summary_df)

    return summary_df