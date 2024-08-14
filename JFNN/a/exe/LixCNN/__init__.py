from JFNN.a.core.Data import Dataset
from JFNN.a.core.Models import CNN
from JFNN.a.core.Utils import Training
from JFNN.a.core.Utils import Testing


class LixCNN(CNN):
    def __init__(self):
        super().__init__()
        self.to("cuda")
        self.__dataset: Dataset | None = None
        self.__training: Training | None = None
        self.__testing: Testing | None = None
        self.setup_dataset()
        self.__create_instaces()

    def setup_dataset(self, dataset_type: str = "CIFAR", is_from_internet: bool = True):
        self.__dataset: Dataset = Dataset(dataset_type=dataset_type, is_from_internet=is_from_internet)

    def __create_testing_instance(self):
        self.__testing = Testing(self, self.__dataset)

    def __create_training_method_instance(self):
        self.__training: Training = Training(self, self.__dataset)

    def __create_instaces(self):
        self.__create_training_method_instance()
        self.__create_testing_instance()

    def dataset_size(self):
        return self.__dataset.trainset_size(), self.__dataset.testset_size()

    def train_model(self, epochs: int = 10):
        self.__training.train(epochs)

    def test_model(self):
        self.__testing.test()
