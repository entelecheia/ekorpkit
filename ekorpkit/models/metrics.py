from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def evaluate_classification_performance(
    true_labels, predicted_labels, average="weighted", labels=None, **kwargs
):
    print("Accuracy: ", accuracy_score(true_labels, predicted_labels))
    print("Precison: ", precision_score(true_labels, predicted_labels, average=average))
    print("Recall: ", recall_score(true_labels, predicted_labels, average=average))
    print("F1 Score: ", f1_score(true_labels, predicted_labels, average=average))
    print(
        "Model Report: \n___________________________________________________",
    )
    print(classification_report(true_labels, predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    return cm
