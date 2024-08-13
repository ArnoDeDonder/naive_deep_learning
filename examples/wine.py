import csv

import numpy as np

from nn.activation_functions import ActivationFunction
from nn.layers import InputLayer, FullyConnectedLayer
from nn.loss_functions import mse_loss


# Get data

LABEL_NAME = 'quality'

with open('datasets/wine.csv', mode='r', newline='') as file:
    wine_data = list(csv.DictReader(file))

feature_names = [feature_name for feature_name, _ in wine_data[0].items() if feature_name != LABEL_NAME]
features_amount = len(feature_names)

print(f'Feature names: {", ".join(feature_names)}.\n'
      f'Label name: {LABEL_NAME}.\n'
      f'Features amount: {features_amount}.')


# Process data
X = np.array(
    [
        [
            float(feature_value)
            for feature_name, feature_value in wine_datapoint.items()
            if feature_name != LABEL_NAME
        ]
        for wine_datapoint in wine_data
    ]
)

y = np.array(
    [
        [
            float(feature_value)
            for feature_name, feature_value in wine_datapoint.items()
            if feature_name == LABEL_NAME
        ]
        for wine_datapoint in wine_data
    ]
)

print(f'Input data shape: {X.shape}\nLabel data shape: {y.shape}')


# Build model

input_layer = InputLayer(size=features_amount)
fully_connected_layer_1 = FullyConnectedLayer(
    input_layer=input_layer, activation_function=ActivationFunction.RELU, size=5
)
model = FullyConnectedLayer(
    input_layer=fully_connected_layer_1, activation_function=ActivationFunction.RELU, size=1
)


# Test model

output = model(X[:1])
print(f"Test output model: {output[0][0]}")


# Test loss function

pred = model(X)
loss = mse_loss(y, pred)
print(f'Initial loss on entire dataset: {loss}')
