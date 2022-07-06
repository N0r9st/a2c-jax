from abc import ABC, abstractmethod

class BaseBuffer(ABC):
    @abstractmethod
    def __init__(self, size: int, *args, **kwargs):
        pass
    def add_experienve