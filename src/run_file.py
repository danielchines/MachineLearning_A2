import mnist_loader
import network
import time

t = time.clock()
training_data, training_check, validation_data, test_data = mnist_loader.load_data_wrapper()

#Instantiates a network with x neurons in first layer, y neurons in second layer, z in 3rd
#net = network.Network([x, y, z])
#x and z are hidden layers with x as input and z as output
net = network.Network([784, 30, 10])

#Training data is list of 50,000 tuples as specified in load_data_wrapper
#Training check is reformatted training data with the second entry (unit vector)
    #replaced by the converted digit
#validation_data and test_data is list of 10,000 2 tuples (x,y) with x as 784D numpy.ndarray
    #and y as corresponding classification digit (integer)

#SGD(tdata, tcheck, vdata, epochs, mini-batch size, learning rate, testData = none->if nothing in test_data)
net.SGD(training_data, training_check, validation_data, 30, 10, 3.0, test_data=test_data)
print time.clock(), "seconds process time"
