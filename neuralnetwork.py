import random
import numpy as np


class NeuralNetwork(object):
    def __init__(self, structure):
        """
        Set initial network's initial structure, biases and weights.
        :params structure: How the is the network's structure?
        """
        # Set network's structure.
        self.structure = structure # e.g [784, 30, 10]
        self.num_layers = len(self.structure) # e.g 3
        # Set weights and biases randomly. Biases are assigned to neurons
        # of all the layers except the first one. Weights, however, are
        # assigned to all connections among nodes.
        hidden_and_output_layers = self.structure[1:]
        input_and_hidden_layers = self.structure[:-1]
        # e.g [<first_hidden_layer>, ..., <output_layer>]
        self.biases = [np.random.randn(layer, 1) for layer in hidden_and_output_layers]
        # e.g [<first_links_between_input_and_hidden>, ..., <links_between_hidden_and_output>]
        self.weights = [np.random.randn(y, x) for x, y in
            zip(input_and_hidden_layers, hidden_and_output_layers)]

    def SGD(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        """
        Train the neural network using stochastic gradient descent.
        :params training_data: Tuple (x, y) being x the network input and
                               y the network output.
        :params epochs: How many iterations we want to perform.
        :params batch_size: How many inputs include each batch.
        :params learning_rate: How fast is learning the network?
        """
        training_data_size = len(training_data)

        for epoch in xrange(epochs):
            random.shuffle(training_data)
            batches = [training_data[start:start+batch_size]
                for start in xrange(0, training_data_size, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if test_data:
                test_data_size = len(test_data)
                print "Epoch {0}: {1} / {2}".format(
                    epoch, self.evaluate(test_data), test_data_size)

    def update_batch(self, batch, learning_rate):
        """
        Update the network's weights and biases using gradient descent
        (backpropagation method).

        :params batch: Each pack containing several network inputs.
        :params learning_rate: How fast is learning the network?
        """
        biases = [np.zeros(biase.shape) for biase in self.biases]
        weights = [np.zeros(weight.shape) for weight in self.weights]

        for network_input, network_output in batch:
            deltas_biase, deltas_weight = self.backpropagation(
                network_input, network_output)
            biases = [new_biase + delta for new_biase, delta in zip(biases, deltas_biase)]
            weights = [new_weight + delta for new_weight, delta in zip(weights, deltas_weight)]

        self.weights = [weight - (learning_rate/len(batch)) * new_weight
            for weight, new_weight in zip(self.weights, weights)]
        self.biases = [biase - (learning_rate/len(batch)) * new_biase
            for biase, new_biase in zip(self.biases, biases)]

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def backpropagation(self, network_input, network_output):
        biases = [np.zeros(biase.shape) for biase in self.biases]
        weights = [np.zeros(weight.shape) for weight in self.weights]
        # The network input can be considered as the first activation values.
        activation_value = network_input
        activations_per_layer = [network_input]
        weighted_outputs = []

        # First phase: Feedforward
        # In this first phase we want to calculate actual output based on the
        # current biases and weights. After that, we can compare this value
        # with the desired one. This will give us a total cost based on the
        # cost function. In the next phase we'll calculate which part of this
        # cost corresponds to which neuron.
        for biase, weight in zip(self.biases, self. weights):
            weighted_output = np.dot(weight, activation_value) + biase
            weighted_outputs.append(weighted_output)
            activation_value = self.sigmoid(weighted_output)
            activations_per_layer.append(activation_value)

        output_layer = -1
        last_hidden_layer = -2
        activations_output_layer = activations_per_layer[output_layer]
        # If the cost is high, then delta is high as well.
        # This means that the new weights and biases will be
        # heavily updated. The reasoning behind is that when
        # there's a big error, a big change is required as well.
        # Also, when a slope is close to Zero, this means that
        # the network is quite confident, so why should we change
        # much anyway?
        delta = ((activations_output_layer - network_output) *
            self.sigmoid_prime(weighted_outputs[output_layer]))

        biases[output_layer] = delta
        weights[output_layer] = np.dot(
                delta,
                activations_per_layer[last_hidden_layer].transpose())
        # Second phase: Backward
        # In this second phase we want to go backwards and assign
        # a gradient for each biase and weight in each layer. This
        # way, we can start modifying these values in order to
        # optimize the network.

        for layer in xrange(2, self.num_layers):
            next_layer = -layer + 1
            actual_layer = -layer
            previous_layer = -layer - 1

            weighted_output = weighted_outputs[actual_layer]
            # We use the previous calculated delta because we want to know how
            # much each neuron contributes to the final error.
            delta = (np.dot(self.weights[next_layer].transpose(), delta)
                * self.sigmoid_prime(weighted_output))
            biases[-layer] = delta
            weights[-layer] = np.dot(delta, activations_per_layer[previous_layer].transpose())

        return (biases, weights)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
