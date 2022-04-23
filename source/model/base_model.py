import abc


class BaseModel(metaclass=abc.ABCMeta):

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build()

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'build')
                and callable(subclass.build)
                or hasattr(subclass, 'get_model')
                and callable(subclass.get_model)
                or NotImplemented)

    @abc.abstractmethod
    def build(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model(self):
        return self.model
