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
        self.mean_b = [np.zeros(b.shape) for b in self.biases]
        self.mean_w = [np.zeros(w.shape) for w in self.weights]
        self.variance_b = [np.zeros(b.shape) for b in self.biases]
        self.variance_w = [np.zeros(w.shape) for w in self.weights]

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
                state = 1
                self.update_mini_batch(mini_batch, eta, state)
            if test_data:
                print("Epoch {}: {} / {} = {}%".format(j, self.evaluate(test_data), n_test, int(self.evaluate(test_data))*100/int(n_test)))
                #print("Epoch {}: {} / {}, Acertos: {}".format(j, self.evaluate(test_data), n_test, self.acertos(test_data)))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta, state):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        theta_b, theta_w = self.adam(nabla_b, nabla_w, state, eta)
        self.weights = [w - nw
                        for w, nw in zip(self.weights, theta_w)]
        self.biases = [b - nb
                       for b, nb in zip(self.biases, theta_b)]
        
    def adam(self, grad_b, grad_w, state, eta):
        beta_1 = 0.9
        beta_2 = 0.999
        eps = 10**(-8)
        newg_b = []
        newg_w = []
        i=0
        j=0
        for (g_b, m_b, v_b) in zip(grad_b, self.mean_b, self.variance_b):
            m_b = beta_1 * m_b + (1 - beta_1)*g_b
            m_b_hat = m_b / (1 - beta_1**(state))
            v_b = beta_2 * v_b + (1-beta_2)*(np.square(g_b))
            v_b_hat = v_b / (1 - beta_2**(state))
            quasenewg_b = eta*m_b_hat/(np.sqrt(v_b_hat)+eps)
            newg_b.append(quasenewg_b)
            self.mean_b[i] = m_b
            self.variance_b[i] = v_b
            i += 1
        for (g_w, m_w, v_w) in zip(grad_w, self.mean_w, self.variance_w):
            m_w = beta_1 * m_w + (1 - beta_1)*g_w
            m_w_hat = m_w / (1 - beta_1**(state))
            v_w = beta_2 * v_w + (1-beta_2)*(np.square(g_w))
            v_w_hat = v_w / (1 - beta_2**(state))
            quasenewg_w = eta*m_w_hat/(np.sqrt(v_w_hat)+eps)
            newg_w.append(quasenewg_w)
            self.mean_w[j] = m_w
            self.variance_w[j] = v_w
            j += 1
        state += 1
        return (newg_b, newg_w)

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
training_data_RU, test_data_RU = data_loader.load_data_todos("RU")

net_RU = Network([4,20,20,10])
net_RU_a = Network([4,20,20,10])
net_RU_j = Network([4,20,20,10])
net_RS_a = Network([4,20,20,10])
net_RS_j = Network([4,20,20,10])
net_RA_a = Network([4,20,20,10])
net_RA_j = Network([4,20,20,10])

#print("RU todos")
#net_RU.SGD(training_data_RU, 20, 20, 0.75, test_data=test_data_RU)
print("Almoço RU")
net_RU_a.SGD(training_data_almoco_RU, 50, 15, 0.001, test_data=test_data_almoco_RU)
print("Janta RU")
net_RU_j.SGD(training_data_janta_RU, 50, 15, 0.001, test_data=test_data_janta_RU)

print("Almoço RS")
net_RS_a.SGD(training_data_almoco_RS, 50, 15, 0.001, test_data=test_data_almoco_RS)

print("Janta RS")
net_RS_j.SGD(training_data_janta_RS, 50, 15, 0.001, test_data=test_data_janta_RS)

print("Almoço RA")
net_RA_a.SGD(training_data_almoco_RA, 50, 15, 0.001, test_data=test_data_almoco_RA)

print("Janta RA")
net_RA_j.SGD(training_data_janta_RA, 50, 15, 0.001, test_data=test_data_janta_RA)

