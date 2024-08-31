from abc import ABC, abstractmethod
from typing_extensions import TypeVar, TypeVarTuple, Generic, Unpack

Input = TypeVarTuple("Input")
Output = TypeVar("Output")


class Trainer(ABC, Generic[Unpack[Input], Output]):

    @abstractmethod
    def step(self, *args: Unpack[Input]) -> Output:
        raise NotImplementedError