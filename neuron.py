from typing import List, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod

import numpy as np


class AbstractNeuron(ABC):
    def __call__(self, input_data: np.ndarray[float]):
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def input(self) -> Union[np.ndarray[float], List['AbstractNeuron']]:
        raise NotImplementedError


class InputNeuron(AbstractNeuron):
    def __init__(self, index: int):
        self._index = index

    def __call__(self, input_data: np.ndarray[float]):
        return input_data[:, self._index]

    @property
    def size(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def input(self) -> Union[np.ndarray[float], List['AbstractNeuron']]:
        raise RuntimeError('InputNeuron does not have input neurons.')


class BasicNeuron(AbstractNeuron):
    def __init__(self, inputs: Optional[List[AbstractNeuron]], activation_function: Callable, size: Optional[int] = None):
        if size is None:
            if inputs is None or len(inputs) < 1:
                raise ValueError('When no input neuron are provided, a size is obligatory.')
            size = len(inputs)

        self._input_neurons: Optional[List[AbstractNeuron]] = inputs
        self.weights: np.ndarray[float] = np.random.rand(size)
        self.bias: float = np.random.rand(1)
        self.activation_function: Callable = activation_function

    def __call__(self, input_data: np.ndarray[float]):
        if self.input() is not None and len(self.input()) > 0:
            input_data = np.stack([neuron(input_data) for neuron in self.input()], axis=-1)
        product_with_bias = input_data @ self.weights + self.bias
        return self.activation_function(product_with_bias)

    @property
    def size(self) -> Tuple[int, ...]:
        return (len(self.weights),)

    def input(self) -> List[AbstractNeuron]:
        return self._input_neurons



