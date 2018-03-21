import matplotlib.pyplot as plt
import numpy as np

def show_confusion_matrix(cnf_matrix, class_labels):
    plt.matshow(cnf_matrix, cmap=plt.cm.YlGn, alpha=0.7)
    ax = plt.gca()
    ax.set_ylabel("Actual Label", fontsize=16, rotation=90)
    ax.set_yticks(range(0, len(class_labels)))
    ax.set_yticklabels(class_labels, rotation=90)
    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_xticks(range(0, len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for row in range(len(cnf_matrix)):
        for col in range(len(cnf_matrix[row])):
            ax.text(col, row, cnf_matrix[row][col], va='center', ha='center', fontsize=16)
    plt.show()


class SKlearnHelper(object):
    def __init__(self, clf, seed, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)