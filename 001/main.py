from Models import CNN
from Data import Dataset
from Utils import Training
if __name__ == '__main__':
    model = CNN()
    print(model)
    print(model.get_summary())
    dataset = Dataset()
    train = Training(model, dataset)
    print(f"Trainset lenght: {len(dataset.trainset)}")
    print(f"Testset lenght: {len(dataset.testset)}")

    train.train(10)
