from JFNN.a.exe import LixCNN
if __name__ == '__main__':
    model = LixCNN()
    print(model)
    print(model.get_summary())
    print(f"Trainset lenght: {model.dataset_size()[0]}")
    print(f"Testset lenght: {model.dataset_size()[1]}")
    model.train_model(10)
    model.test_model()
