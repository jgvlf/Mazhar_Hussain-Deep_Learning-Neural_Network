from JFNN.a.core.Data import Dataset
from JFNN.a.core.Models import CNN
from JFNN.a.core.Utils import Training
from JFNN.a.core.Utils import Testing
from JFNN.a.core.Utils import Evaluation


class LixCNN(CNN):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.to("cuda")
        self.__dataset: Dataset = dataset
        self.__training: Training | None = None
        self.__testing: Testing | None = None
        self.__evaluation: Evaluation | None = None
        self.__create_instaces()
        
    def __create_instaces(self) -> None:
        self.__create_training_method_instance()
        self.__create_testing_instance()
        self.__create_evaluation_instance()

    def __create_testing_instance(self) -> None:
        self.__testing = Testing(self, self.__dataset)

    def __create_training_method_instance(self) -> None:
        self.__training: Training = Training(self, self.__dataset)

    def __create_evaluation_instance(self):
        self.__evaluation = Evaluation(self, self.__dataset)
    
    def dataset_size(self) -> tuple[int, int]:
        return self.__dataset.trainset_size(), self.__dataset.testset_size()

    def train_model(self, epochs: int = 10) -> None:
        self.__training.train(epochs)

    def test_model(self) -> None:
        self.__testing.test()

    def confusion_matrix(self, show: bool, save: bool):
        self.__evaluation.confusion_matrix(show, save)

    def seaborn_confusion_matrix(self, show: bool, save: bool):
        self.__evaluation.seaborn_confusion_matrix(show, save)

    def classification_report(self):
        self.__evaluation.class_report()