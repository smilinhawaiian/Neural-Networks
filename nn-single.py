# Sharice Mayer
# 04-16-20
# CS445 ML - Spring 2020
# HW 1 #11
# Create a single-layer perceptron NN!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
# import random
# import math


class Perceptron:
    # constructor here
    def __init__(self, input_size, momentum, learning_rate, my_label):   # first parameter of any class method is self
        self.input_size = input_size  # number of inputs - x_i in each example
        self.momentum = momentum
        self.learning_rate = learning_rate  # learning rate == step size
        self.my_label = my_label  # target label
        # create random  initial weight values matrix
        self.weight_vector = np.random.uniform(low=(-.05), high=.05, size=(input_size+1,))  #+1 for bias incl.

    # train a single example(input_vector)
    def train(self, input_vector, target):  # target==0 if expected label == perceptron label
        # forward propagation
        # insert bias (input, indice, valueinserted, axis-optional
        input_vector = np.insert(input_vector, 0, 1, axis=0)  # is bias 1/255 or 1?
        # TODO: Compare compute activation to compute below
        # compute activation function
        # vector_dot = np.dot(self.weight_vector, input_vector)  # dot product
        # output = (1 / (1 + (math.exp(-(vector_dot)))))  # sigmoid
        # return output
        # compute
        sum_value = 0
        if np.dot(self.weight_vector, input_vector) > 0:
            # update weight vector
            sum_value = 1
        # compare to target
        if sum_value != target:
            # update weights taking in target and value with inline anonymous function
            update_funct = lambda w, x_in: w + x_in*(self.learning_rate * (target - sum_value))
            # update a single weight
            vector_function = np.vectorize(update_funct)  # can take in lambda or function
            # does the vector function on the entire vector
            self.weight_vector = vector_function(self.weight_vector, input_vector)

    # test an example
    def test(self, input_vector):
        # insert bias
        input_vector = np.insert(input_vector, 0, 1, axis=0)
        return np.dot(self.weight_vector, input_vector)


class TypeClassifier:
    def __init__(self, input_size, momentum, learning_rate, num_classes):
        self.perceptron_list = []  # create a list of perceptrons with their labels
        for index in range(num_classes):
            # create perceptron labels using index
            self.perceptron_list.append(Perceptron(input_size, momentum, learning_rate, index))

    # train a single example
    def train_classifier_vector(self, input_vector, label):
        # is it labeled correctly
        for perceptron in self.perceptron_list:
            # compute target inline
            target = 1 if label == perceptron.my_label else 0
            perceptron.train(input_vector, target)

    # train entire dataset
    def train_classifier_dataset(self, input_matrix, label_vector):
        # shuffle list
        zipped_data = list(zip(input_matrix, label_vector))
        np.random.shuffle(zipped_data)
        # for each row
        for input_vector, label in zipped_data:
            self.train_classifier_vector(input_vector, label)

    # return a predicted label
    def classify_vector(self, input_vector):
        # list comprehension--
        # predictions = [(perceptron.test(input_vector), perceptron.my_label) for perceptron in self.perceptron_list]
        predictions = [(perceptron.test(input_vector)) for perceptron in self.perceptron_list]
        return np.argmax(predictions)  # argmax returns position of largest val, which will be == label

    # return classification list given input array
    def classify_dataset(self, input_matrix):  # classify entire dataset
        predictions = [(self.classify_vector(input_vector)) for input_vector in input_matrix]
        return predictions

    # return accuracy over dataset
    def get_accuracy(self, predictions, labels):
        correct = 0
        total = 0
        comparison_list = zip(predictions, labels)
        for prediction, label in comparison_list:
            if int(prediction) == int(label):
                correct += 1
            total += 1
        # scale the accuracy data to percent for plotting
        return (correct/total)*100.00


# given a start time, print block time information
def get_time(start_time, end_time, name, optional_num):

    current_time = time.time()
    elapsed = end_time - start_time
    if elapsed == 0:
        elapsed = current_time - start_time
    n_sec = (elapsed % 60)
    n_min = elapsed / 60
    if optional_num < 0:
        print("%s run time: %d minutes, %d seconds\n" % (name, n_min, n_sec))
    else:
        print("%s %d run time: %d minutes, %d seconds" % (name, optional_num, n_min, n_sec))


# read data
# split in to labels and input_matrix
# give to train_classifier

# python for main method:
if __name__ == "__main__":

    # Print Header information
    template = "{0:14}{1:20}"
    print("")
    print(template.format("Course:", "CS445-ML"))
    print(template.format("Date:", "04/17/20"))
    print(template.format("Name:", "Sharice Mayer"))
    print(template.format("Assignment:", "HW1"))
    print(template.format("Topic:", "NN-Single-Layer-Perceptrons"))
    print("")

    # Start a timer for program
    start = time.time()

    # Create matrices of data
    train_data = pd.read_csv("data/mnist_train.csv", header=None).values
    test_data = pd.read_csv("data/mnist_test.csv", header=None).values
    # train_data = pd.read_csv("data/tiny_train.csv", header=None).values
    # test_data = pd.read_csv("data/tiny_test.csv", header=None).values
    # train_data = pd.read_csv("data/3k_train.csv", header=None).values
    # test_data = pd.read_csv("data/3k_test.csv", header=None).values

    # create perceptron labels
    train_labels = train_data[:, 0]   # create labels vector for train perceptrons
    test_labels = test_data[:, 0]     # create labels vector for test perceptrons

    # delete the first column (labels) for data computation
    train_data = np.delete(train_data, 0, axis=1)
    test_data = np.delete(test_data, 0, axis=1)

    # pre-process - scale data to 0 to 1 from 0 to 255
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # train perceptrons with 3 learning rates - η = 0.001, 0.01, 0.1
    # learning_rates = [0.001, 0.010, 0.100]  # step_size
    # plot_num = 0            # plot number for graphs
    # learning_rate = 0.001   # For testing
    # plot_num = 1          # plot number for graphs
    # learning_rate = 0.01  # For testing
    plot_num = 2          # plot number for graphs
    learning_rate = 0.1   # For testing
    momentum = 0.9  # what was default momentum?
    num_inputs = 784  # 28x28 pixels per instance
    num_classes = 10  # 0-9 perceptrons
    train_accuracy = []  # for plotting
    test_accuracy = []  # for plotting

    # create a classifier object -  initialise weights and perceptrons
    digit_classifier = TypeClassifier(num_inputs, momentum, learning_rate, num_classes)

    # calculate accuracy on test and train sets before training:
    train_predictions = digit_classifier.classify_dataset(train_data)
    test_predictions = digit_classifier.classify_dataset(test_data)
    train_accuracy.append(digit_classifier.get_accuracy(train_predictions, train_labels))
    test_accuracy.append(digit_classifier.get_accuracy(test_predictions, test_labels))

    # train for 70 epochs
    epochs = 1    # 70
    for epoch in range(epochs):  # epoch = 0 to 69
        # epoch_start = time.time()

        # train perceptrons with training dataset
        digit_classifier.train_classifier_dataset(train_data, train_labels)

        # calculate accuracy after training
        train_predictions = digit_classifier.classify_dataset(train_data)
        test_predictions = digit_classifier.classify_dataset(test_data)
        train_accuracy.append(digit_classifier.get_accuracy(train_predictions, train_labels))
        test_accuracy.append(digit_classifier.get_accuracy(test_predictions, test_labels))

        # if accuracy isn't changing  anymore, no need to continue testing
        #if abs(train_accuracy[epoch+1] - train_accuracy[epoch]) < 0.01:
        #    print(f"No change, so end at epoch = {epoch}")

        # print time for single epoch
        # epoch_end = time.time()
        # get_time(epoch_start, epoch_end, "Epoch", epoch+1)
        # print total running time
        # current = time.time()
        # get_time(start, start, "Total current", -1)

    print(f"\nFinal number of epochs = {epochs}")
    # confusion matrix for test data after training
    print(f"Confusion Matrix for learning rate: {learning_rate}:")
    print(confusion_matrix(test_labels, test_predictions))

    # create a plot of the data accuracy
    plot_num += 1
    plt.figure(figsize=(10, 10))
    plt.axis([0, epochs, 0, 100])  # v = [xmin, xmax, ymin, ymax]
    plt.grid(True)
    plt.title('{0}_Epoch_accuracy_during_MNIST_training_step_size_{1}'.format(epochs, learning_rate))
    x = range(epochs + 1)  # whatever the for-loop is+1
    plt.plot(x, train_accuracy, label='Train Accuracy')
    plt.plot(x, test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.legend(loc=2)
    plt.axis('tight')
    plt.savefig(('NN_MLP_Figure' + str(plot_num) + '_Step' + str(learning_rate) + '.jpg'), bbox_inches='tight')
    # plt.savefig((str(epochs) + 'Epoch_Figure' + str(plot_num) + '_Step' + str(learning_rate) + '.jpg'))

    # print total program time
    end = time.time()
    get_time(start, end, "\nTotal program", -1)
    print("")
