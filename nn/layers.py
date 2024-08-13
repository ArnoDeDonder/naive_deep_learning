from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union

import numpy as np

from nn.activation_functions import ActivationFunction
from nn.neurons import AbstractNeuron, InputNeuron, BasicNeuron


class AbstractLayer(ABC):
    def __call__(self, input_data: np.ndarray[float]):
        raise NotImplementedError

    @property
    @abstractmethod
    def neurons(self) -> List[AbstractNeuron]:
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def input(self) -> 'AbstractLayer':
        raise NotImplementedError


class InputLayer(AbstractLayer):
    def __init__(self, size: int):
        self._neurons = [InputNeuron(index=i) for i in range(size)]

    def __call__(self, input_data: np.ndarray[float]):
        return input_data

    @property
    def neurons(self) -> List[AbstractNeuron]:
        return self._neurons

    @property
    def size(self) -> Tuple[int, ...]:
        return (len(self.neurons),)

    @property
    def input(self) -> AbstractLayer:
        raise RuntimeError('InputLayer does not have input layers.')


class FullyConnectedLayer(AbstractLayer):
    def __init__(self, input_layer: AbstractLayer, size: int, activation_function: Callable = ActivationFunction.RELU):
        self._input_layer: AbstractLayer = input_layer
        self.activation_function: Callable = activation_function
        self._neurons: List[AbstractNeuron] = [
            BasicNeuron(inputs=self._input_layer.neurons, activation_function=self.activation_function)
            for _ in range(size)
        ]

    def __call__(self, input_data: np.ndarray[float]):
        return np.stack([neuron(input_data) for neuron in self.neurons], axis=-1)

    @property
    def neurons(self) -> List[AbstractNeuron]:
        return self._neurons

    @property
    def size(self) -> Tuple[int, ...]:
        return (len(self.neurons),) + self._input_layer.size

    @property
    def input(self) -> AbstractLayer:
        return self._input_layer



