import numpy as np

from nn.activation_functions import ActivationFunction
from nn.neurons import BasicNeuron
from nn.layers import InputLayer, FullyConnectedLayer

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

    print(f'Size of top perceptron (= amount of weights): {perceptron_4.size}.')
    print('Output of top perceptron with given input data:')
    print(perceptron_4(input_data=input_data), end='\n\n')

    # Simple fully connected layers

    input_layer = InputLayer(size=4)
    fully_connected_layer_1 = FullyConnectedLayer(
        input_layer=input_layer, activation_function=ActivationFunction.RELU, size=3
    )
    fully_connected_layer_2 = FullyConnectedLayer(
        input_layer=fully_connected_layer_1, activation_function=ActivationFunction.RELU, size=5
    )

    print(f'Size of top fully connected layer (= amount of neurons in this + lower layers): '
          f'{fully_connected_layer_2.size}.')
    print('Output of top layer with given input data:')
    print(fully_connected_layer_2(input_data=input_data), end='\n\n')
