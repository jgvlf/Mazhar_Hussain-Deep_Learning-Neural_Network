from JFNN.a.exe import LixCNN
from JFNN.a.core.Data import Dataset
if __name__ == '__main__':
    dataset = Dataset("CIFAR", True, "./data")
    model = LixCNN(dataset)
    print(model)
    print(model.get_summary())
    print(f"Trainset lenght: {model.dataset_size()[0]}")
    print(f"Testset lenght: {model.dataset_size()[1]}")
    model.train_model(10)
    model.test_model()
    model.confusion_matrix(False, True)
    model.seaborn_confusion_matrix(False, True)
    model.classification_report()