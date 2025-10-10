"""
network.py
~~~~~~~~~~

Módulo de implementação de rede neural por modelo de gradiente estocástico em mini-batch.

"""

#### Bibliotecas
import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """O parâmetro ``sizes`` deve ser uma lista contendo o número
        de neurônios em cada camada. Por exemplo, a lista [3, 10, 1] representa
        uma rede neural com 3 entradas, 10 neurônios na primeira camada oculta e 
        1 neurônio de saída. Os pesos (weights) e viéses (biases) são inicializados
        aleatoriamente em distribuição normal."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Função que retorna a saída de uma entrada ``a`` na rede neural."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Função responsável por treinar os pesos e viéses por gradiente estocástico
        em mini-batch. Perceba que após cada época, é mostrado como a rede neural
        esta performando em um conjunto de dados teste (test_data)."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {}: {} / {} = {}%".format(j, self.evaluate(test_data), n_test, int(self.evaluate(test_data))*100/int(n_test)))
                #print("Epoch {}: {} / {}, Acertos: {}".format(j, self.evaluate(test_data), n_test, self.acertos(test_data)))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = [x]
        activations = [np.array([x]).transpose()] # lista que acumula as activations em cada camada
        zs = [] # lista que acumula as ativações sem a função de ativação
        for b, w in zip(self.biases, self.weights):
            activation = np.array(activation).transpose()
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            activation = np.array(activation).transpose()
        # deltas da última camada
        delta = self.cost_derivative(np.array(activations[-1]).transpose(), y) * np.array(sigmoid_prime(zs[-1])).transpose()
        nabla_b[-1] = np.array(delta).transpose()
        nabla_w[-1] = np.dot(np.array(delta).transpose(), np.array(activations[-2]).transpose())
        # deltas das camadas anteriores
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(np.array(self.weights[-l+1]).transpose(), np.array(delta).transpose()) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.array(activations[-l-1]).transpose())
            delta = np.array(delta).transpose()
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(np.array([x]).transpose())), np.argmax(y))
                        for (x, y) in test_data]
        return sum(x == y for (x, y) in test_results)
    
    def acertos(self, test_data):
        test_results = [(x, np.argmax(self.feedforward(np.array([x]).transpose())), np.argmax(y))
                        for (x, y) in test_data]
        acertos = []
        for (x,x_f,y) in test_results:
            if x_f == y:
                acertos.append(x)
        return acertos

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#### Funções de Ativação
def sigmoid(z):
    sigmoid_z = [1.0/(1.0+np.exp(-x)) for x in z]
    return sigmoid_z

def sigmoid_prime(z):
    sigmoid_prime_z = []
    for x in z:
        sigmoid_prime_z.append(1.0/(1.0+np.exp(-x))*(1 - 1.0/(1.0+np.exp(-x))))
    return sigmoid_prime_z

"""Implementação"""

import data_loader
training_data_almoco_RU, test_data_almoco_RU, training_data_janta_RU, test_data_janta_RU = data_loader.load_data("RU")
training_data_almoco_RS, test_data_almoco_RS, training_data_janta_RS, test_data_janta_RS = data_loader.load_data("RS")
training_data_almoco_RA, test_data_almoco_RA, training_data_janta_RA, test_data_janta_RA = data_loader.load_data("RA")

net = Network([4,20,20,20,10])
print("Almoço RU")
net.SGD(training_data_almoco_RU, 6, 50, 4.0, test_data=test_data_almoco_RU)
print("Janta RU")
net.SGD(training_data_janta_RU, 6, 50, 4.0, test_data=test_data_janta_RU)

print("Almoço RS")
net.SGD(training_data_almoco_RS, 6, 50, 5.0, test_data=test_data_almoco_RS)
print("Janta RS")
net.SGD(training_data_janta_RS, 6, 50, 5.0, test_data=test_data_janta_RS)

print("Almoço RA")
net.SGD(training_data_almoco_RA, 6, 50, 5.0, test_data=test_data_almoco_RA)
print("Janta RA")
net.SGD(training_data_janta_RA, 6, 50, 5.0, test_data=test_data_janta_RA)

