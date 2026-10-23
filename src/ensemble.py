import data_loader
import network
import svm
import numpy as np

def ensemble(test_data_almoco, test_data_janta):
    test_data_clf_0_a = []
    test_data_clf_1_a = []
    test_data_clf_0_j = []
    test_data_clf_1_j = []
    for i in range(len(test_data_almoco)):
        test_data_clf_0_a.append(test_data_almoco[i][0])
        test_data_clf_1_a.append(test_data_almoco[i][1])
        test_data_clf_0_j.append(test_data_janta[i][0])
        test_data_clf_1_j.append(test_data_janta[i][1])
    test_results_net_a = [(np.argmax(network.net_RU_a.feedforward(np.array([x]).transpose())))
                        for (x, y) in test_data_almoco]
    test_results_net_j = [(np.argmax(network.net_RU_j.feedforward(np.array([x]).transpose())))
                        for (x, y) in test_data_janta]
    test_results_svm_a = [int(a) for a in svm.clf.predict(test_data_clf_0_a)]
    test_results_svm_j = [int(a) for a in svm.clf.predict(test_data_clf_0_j)]
    test_results_ensemble_a = [(result_net + 2*result_clf)/3 for result_net, result_clf in zip(test_results_net_a, test_results_svm_a)]
    test_results_ensemble_j = [(2*result_net + result_clf)/3 for result_net, result_clf in zip(test_results_net_j, test_results_svm_j)]
    num_correct_a = sum(int(np.floor(a) == np.argmax(y)) for a, y in zip(test_results_ensemble_a, test_data_clf_1_a))
    num_correct_j = sum(int(np.floor(a) == np.argmax(y)) for a, y in zip(test_results_ensemble_j, test_data_clf_1_j))
    num_correct_clf_a = sum(int(a == np.argmax(y)) for a, y in zip(test_results_svm_a, test_data_clf_1_a))
    num_correct_clf_j = sum(int(a == np.argmax(y)) for a, y in zip(test_results_svm_j, test_data_clf_1_j))
    print("{} of {} values correct = {}%".format(num_correct_clf_a, len(test_data_clf_0_a), num_correct_clf_a*100/len(test_data_clf_1_a)))
    print("{} of {} values correct = {}%".format(num_correct_clf_j, len(test_data_clf_0_j), num_correct_clf_j*100/len(test_data_clf_1_j)))
    print("Almoço: {} of {} values correct = {}%".format(num_correct_a, len(test_data_clf_0_a), num_correct_a*100/len(test_data_clf_1_a)))
    print("Janta: {} of {} values correct = {}%".format(num_correct_j, len(test_data_clf_0_j), num_correct_j*100/len(test_data_clf_1_j)))

training_data_almoco_RU, test_data_almoco_RU, training_data_janta_RU, test_data_janta_RU = data_loader.load_data("RU")
ensemble(test_data_almoco_RU, test_data_janta_RU)