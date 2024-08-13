from abc import abstractmethod


class HyperParametersOptimization:
    def __init__(self) -> None:
        self.criterion = self.define_loss_function()
        self.optimizer = self.define_optimizer()

    @abstractmethod
    def define_loss_function(self) -> object:
        pass

    @abstractmethod
    def define_optimizer(self) -> object:
        pass
