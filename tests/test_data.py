def test_data():
    dataset = MNIST(...)
    assert len(dataset) == N_train for training and N_test for test
    assert that each datapoint has shape [1,224,224]
    assert that all labels are represented
