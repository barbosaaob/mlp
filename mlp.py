import numpy as np


class Neuron():
    def __init__(self, nweights):
        """
        Neuron constructor.
        f(x) = < x, w > + theta
        """
        self.nweights = nweights
        self.weights = np.zeros(nweights,)
        # initialize weights randomly
        for i in range(nweights):
            if np.random.rand() > 0.5:
                sign = -1
            else:
                sign = 1
            self.weights[i] = sign * np.random.rand()

        # initialize theta randomly
        if np.random.rand() > 0.5:
            sign = -1
        else:
            sign = 1
        self.theta = sign * np.random.rand()

    def __str__(self):
        s = "Neuron weights:"
        for i in range(self.nweights):
            s += " " + str(self.weights[i])
        return s + "\n"

    def set_weights(self, weights):
        """
        Set weights.
        """
        self.weights = weights

    def get_weights(self):
        """
        Get weights.
        """
        return self.weights

    def get_weight_i(self, i):
        """
        Get weight at position i.
        """
        return self.weights[i]

    def set_theta(self, theta):
        """
        Set theta.
        """
        self.theta = theta

    def get_theta(self):
        """
        Get theta.
        """
        return self.theta

    def transfer_func(self, x):
        """
        Computes transfer function.
        """
        # sigmoid function
        r = 1.0 / (1.0 + np.exp(-x))

        return r

    def output(self, input):
        """
        Computes neuron output.
        """
        net = 0.0
        for i in range(self.nweights):
            net += input[i] * self.weights[i]
        net += self.theta
        f_net = self.transfer_func(net)
        return f_net


class MLP(Neuron):
    def __init__(self, input_len, hidden_len=2, output_len=1, eta=1e-2,
                 eps=1e-6):
        self.eta = eta
        self.eps = eps
        self.input_len = input_len
        self.hidden_len = hidden_len
        self.output_len = output_len
        self.hidden_layer = list()
        self.output_layer = list()
        # initialize hidden layer
        for i in range(hidden_len):
            self.hidden_layer.append(Neuron(input_len))
        # initialize output layer
        for i in range(output_len):
            self.output_layer.append(Neuron(output_len))

    def backwards(self, data):
        it = 0
        error = self.eps + 1.0
        # enquanto erro for grande
        while error > self.eps:
            error = 0.0
            # para cada instancia de dado
            for k in range(data.shape[0]):
                input = data[k, :self.input_len]
                desired = data[k, -self.output_len:]
                # processando na camada escondida
                hidden = np.zeros(self.hidden_len,)
                for i in range(self.hidden_len):
                    neuron = self.hidden_layer[i]
                    hidden[i] = neuron.output(input)
                # processando na camada de saida
                output = np.zeros(self.output_len,)
                for i in range(self.output_len):
                    neuron = self.output_layer[i]
                    output[i] = neuron.output(hidden)
                    error += (desired[i] - output[i])**2

                # treinamento
                delta_o = np.zeros(self.output_len,)
                for i in range(self.output_len):
                    neuron = self.output_layer[i]
                    df_dnet = output[i] * (1.0 - output[i])
                    delta_o[i] = (desired[i] - output[i]) * df_dnet

                delta_h = np.zeros(self.hidden_len,)
                for i in range(self.hidden_len):
                    delta_h[i] = hidden[i] * (1.0 - hidden[i])
                    sum = 0.0
                    for j in range(self.output_len):
                        neuron = self.output_layer[j]
                        w = neuron.get_weight_i(i)
                        sum += delta_o[j] * w
                    delta_h[i] *= sum

                # atualizar os pesos e os thetas
                # camada de saida
                for i in range(self.output_len):
                    neuron = self.output_layer[i]
                    weights = neuron.get_weights()
                    for j in range(weights.shape[0]):
                        weights[j] = weights[j] + self.eta * delta_o[i] * \
                            hidden[j]
                    theta = neuron.get_theta()
                    theta = theta + self.eta * delta_o[i]
                    neuron.set_weights(weights)
                    neuron.set_theta(theta)

                # camada escondida
                for i in range(self.hidden_len):
                    neuron = self.hidden_layer[i]
                    weights = neuron.get_weights()
                    for j in range(weights.shape[0]):
                        weights[j] = weights[j] + self.eta * delta_h[i] * \
                            input[j]
                    theta = neuron.get_theta()
                    theta = theta + self.eta * delta_h[i]
                    neuron.set_weights(weights)
                    neuron.set_theta(theta)
            it += 1
            if not np.mod(it, 1000):
                print "Error:", error

    def forward(self, data):
        for k in range(data.shape[0]):
            input = data[k, :self.input_len]
            hidden = np.zeros(self.hidden_len,)
            for i in range(self.hidden_len):
                neuron = self.hidden_layer[i]
                hidden[i] = neuron.output(input)

            output = np.zeros(self.output_len,)
            for i in range(self.output_len):
                neuron = self.output_layer[i]
                output[i] = neuron.output(hidden)

            print output


# def test(file):
#     data_set = np.loadtxt(file)
#     data = data_set[:, :-3]
#     data_class = data_set[:, -3]
#     ninst, dim = data.shape
#     train_idx = np.random.permutation(ninst)[:np.ceil(0.7 * ninst)]  # 70%
#     test_idx = np.setdiff1d(range(ninst), train_idx)
#     train = data_set[train_idx, :]
#     test = data[test_idx, :]
#     mlp = MLP(dim, 5, 3, eta=1e-4)
#     mlp.backwards(train)
#     mlp.forward(test)
#     print "real classes:"
#     print data_class[test_idx]


if __name__ == "__main__":
    # test("iris.dat")
    import sys
    assert len(sys.argv) != 7, "Usage: python mlp.py input_len hidden_len \
            output_len train.dat test.dat eta eps"
    input_len = int(sys.argv[1])
    hidden_len = int(sys.argv[2])
    output_len = int(sys.argv[3])
    train_file = sys.argv[4]
    train = np.loadtxt(train_file)
    test_file = sys.argv[5]
    test = np.loadtxt(test_file)
    eta = float(sys.argv[6])
    eps = float(sys.argv[7])
    mlp = MLP(input_len, hidden_len, output_len, eta, eps)
    print "Training..."
    mlp.backwards(train)  # mlp training
    print "Predicting..."
    mlp.forward(test)     # mlp prediction
