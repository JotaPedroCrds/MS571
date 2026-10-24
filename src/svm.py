"""
svm
~~~~~~~~~

Implementação do modelo Support Vector Machine para classificar
intervalo de volumetria de refeição nos restaurantes universitários."""

#### Bibliotecas
# Locais
import data_loader

# Auxiliares
from sklearn import svm
import numpy as np

def svm_baseline(restaurante):
    training_data_almoco, test_data_almoco, training_data_janta, test_data_janta = data_loader.load_data(restaurante)
    treino_almoco_0 = []
    treino_almoco_1 = []
    for i in range(len(training_data_almoco)):
        treino_almoco_0.append(training_data_almoco[i][0])
        treino_almoco_1.append(np.argmax(training_data_almoco[i][1]))
    test_almoco_0 = []
    test_almoco_1 = []
    for j in range(len(test_data_almoco)):
        test_almoco_0.append(test_data_almoco[j][0])
        test_almoco_1.append(np.argmax(test_data_almoco[j][1]))
    treino_janta_0 = []
    treino_janta_1 = []
    for i in range(len(training_data_janta)):
        treino_janta_0.append(training_data_janta[i][0])
        treino_janta_1.append(np.argmax(training_data_janta[i][1]))
    test_janta_0 = []
    test_janta_1 = []
    for j in range(len(test_data_janta)):
        test_janta_0.append(test_data_janta[j][0])
        test_janta_1.append(np.argmax(test_data_janta[j][1]))
    # treinamento
    clf_almoco = svm.SVC(kernel='poly')
    clf_janta = svm.SVC(kernel='poly')
    clf_almoco.fit(treino_almoco_0, treino_almoco_1)
    clf_janta.fit(treino_janta_0, treino_janta_1)
    # teste
    predictions_almoco = [int(a) for a in clf_almoco.predict(test_almoco_0)]
    num_correct_almoco = sum(int(a == y) for a, y in zip(predictions_almoco, test_almoco_1))
    predictions_janta = [int(a) for a in clf_janta.predict(test_janta_0)]
    num_correct_janta = sum(int(a == y) for a, y in zip(predictions_janta, test_janta_1))

    print("Almoço {}".format(restaurante))
    print("{} of {} values correct = {}%".format(num_correct_almoco, len(test_almoco_1), num_correct_almoco*100/len(test_almoco_1)))
    print("Janta {}".format(restaurante))
    print("{} of {} values correct = {}%".format(num_correct_janta, len(test_janta_1), num_correct_janta*100/len(test_janta_1)))
    print(clf_almoco.predict([[1,3,0,3],[1,3,0,3],[1,3,0,1],[5,3,0,3],[5,3,0,3],[3,3,0,3],[2,3,0,3]]))
    print(clf_janta.predict([[1,3,0,3],[1,3,0,3],[5,3,0,1],[5,3,0,3],[3,3,0,3],[2,3,0,3]]))

if __name__ == "__main__":
    svm_baseline("RU")
    svm_baseline("RS")
    svm_baseline("RA")


"""training_data_almoco, test_data_almoco, training_data_janta, test_data_janta = data_loader.load_data("RU")
treino_almoco_0 = []
treino_almoco_1 = []
for i in range(len(training_data_almoco)):
    treino_almoco_0.append(training_data_almoco[i][0])
    treino_almoco_1.append(np.argmax(training_data_almoco[i][1]))
clf = svm.SVC()
clf.fit(treino_almoco_0, treino_almoco_1)"""
