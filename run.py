# How to trigger ANN Mnist example.
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import neuralnetwork as network
net = network.NeuralNetwork([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

