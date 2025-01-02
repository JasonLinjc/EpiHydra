# Copyright (c) Meta Platforms, Inc. and affiliates.
import abc
from typing import Any, Generator, Generic, TypeVar

T = TypeVar("T")
C = TypeVar("C")


class StatefulIterator(Generic[T, C], abc.ABC):

    @abc.abstractmethod
    def get_state(self) -> C:
        pass

    @abc.abstractmethod
    def create_iter(self) -> Generator[T, Any, None]:
        pass


class IteratorState(Generic[C]):
    @abc.abstractmethod
    def build(self) -> StatefulIterator[T, C]:
        pass
