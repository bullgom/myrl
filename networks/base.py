import torch as t
from typing_extensions import TypeVar, TypeVarTuple, Generic, Unpack
from abc import ABC, abstractmethod

Input = TypeVarTuple("Input")
Output = TypeVar("Output")


class Network(t.nn.Module, ABC, Generic[Unpack[Input], Output]):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args: Unpack[Input]) -> Output:
        raise NotImplementedError

    def __call__(self, *args: Unpack[Input]) -> Output:
        return super().__call__(*args)

