from Models import CNN
from Data import Dataset
if __name__ == '__main__':
    model = CNN()
    dataset = Dataset()
    print(f"Trainset lenght: {len(dataset.trainset)}")
    print(f"Testset lenght: {len(dataset.testset)}")
