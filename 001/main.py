from Models import CNN
from Data import Dataset
if __name__ == '__main__':
    model = CNN()
    print(model)
    print(model.get_summary())
    dataset = Dataset()
    print(f"Trainset lenght: {len(dataset.trainset)}")
    print(f"Testset lenght: {len(dataset.testset)}")
