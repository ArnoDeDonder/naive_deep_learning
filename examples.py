import numpy as np

from activation_functions import ActivationFunction
from neuron import BasicNeuron
from layers import InputLayer, FullyConnectedLayer

if __name__ == '__main__':
    input_data = np.array([[.11, .32, .22, .54],
                           [.87, .55, .34, .32],
                           [.23, .44, .34, .21]])

    # Simple Perceptrons

    perceptron_1 = BasicNeuron(inputs=[], size=4, activation_function=ActivationFunction.SIGMOID)
    perceptron_2 = BasicNeuron(inputs=[], size=4, activation_function=ActivationFunction.SIGMOID)
    perceptron_3 = BasicNeuron(inputs=[], size=4, activation_function=ActivationFunction.SIGMOID)

    perceptron_4 = BasicNeuron(inputs=[perceptron_1, perceptron_2, perceptron_3],
                               activation_function=ActivationFunction.SIGMOID)

    print(perceptron_4(input_data=input_data))

    # Simple fully connected layers

    input_layer = InputLayer(size=4)
    fully_connected_layer_1 = FullyConnectedLayer(
        input_layer=input_layer, activation_function=ActivationFunction.RELU, size=5
    )
    fully_connected_layer_2 = FullyConnectedLayer(
        input_layer=fully_connected_layer_1, activation_function=ActivationFunction.RELU, size=5
    )
    print(fully_connected_layer_2(input_data=input_data))

