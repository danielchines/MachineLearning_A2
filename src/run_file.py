import mnist_loader
import network


training_data, training_check, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, training_check, validation_data, 30, 10, 3.0, test_data=test_data)
