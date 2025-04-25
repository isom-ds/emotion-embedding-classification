import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def evaluate(y_true, y_pred_labels, emotions):
    # Create matrix labels
    def create_matrix_labels(labels, emotions=emotions):
        matrix_labels = np.zeros((len(labels), len(emotions)))
        for i, label in enumerate(labels):
            for j, emo in enumerate(emotions):
                if emo in label:
                    matrix_labels[i, j] = 1
        return matrix_labels

    # Reshape
    y_true = create_matrix_labels(y_true)
    y_pred_labels = create_matrix_labels(y_pred_labels)

    # Defining variables
    precision = []
    recall = []
    f1 = []

    # Per emotion evaluation
    idx2emotion = {i: e for i, e in enumerate(emotions)}

    for i in range(len(emotions)):

        # Computing precision, recall and f1-score
        p, r, f1_score, _ = precision_recall_fscore_support(y_true[:, i], y_pred_labels[:, i], average="binary")

        # Append results in lists
        precision.append(round(p, 2))
        recall.append(round(r, 2))
        f1.append(round(f1_score, 2))

    # Macro evaluation
    macro_p, macro_r, macro_f1_score, _ = precision_recall_fscore_support(y_true, y_pred_labels, average="macro")

    # Append results in lists
    precision.append(round(macro_p, 2))
    recall.append(round(macro_r, 2))
    f1.append(round(macro_f1_score, 2))

    # Converting results to a dataframe
    df_results = pd.DataFrame({"Precision":precision, "Recall":recall, 'F1':f1})
    df_results.index = emotions+['MACRO-AVERAGE']

    return df_results