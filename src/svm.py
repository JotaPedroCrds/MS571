"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# My libraries
import data_loader

# Third-party libraries
from sklearn import svm
import numpy as np

def svm_baseline():
    training_data_almoco, test_data_almoco, training_data_janta, training_data_janta = data_loader.load_data('RA')
    treino_0 = []
    treino_1 = []
    for i in range(len(training_data_almoco)):
        treino_0.append(training_data_almoco[i][0])
        treino_1.append(np.argmax(training_data_almoco[i][1]))
    test_0 = []
    test_1 = []
    for j in range(len(test_data_almoco)):
        test_0.append(test_data_almoco[j][0])
        test_1.append(np.argmax(test_data_almoco[j][1]))
    # train
    clf = svm.SVC()
    clf.fit(treino_0, treino_1)
    # test
    predictions = [int(a) for a in clf.predict(test_0)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_1))
    print("Baseline classifier using an SVM.")
    print("{} of {} values correct.".format(num_correct, len(test_1)))  

if __name__ == "__main__":
    svm_baseline()
    
