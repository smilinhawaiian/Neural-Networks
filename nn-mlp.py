# Sharice Mayer
# 04/25/20
# Machine Learning
# Multi-layer-perceptron NN with 1 hidden layer
# Feed-Forward, Back-Propagation, SSE
# Stochastic Gradient Descent, Sigmoid Activation
# Classify Handwritten Digits using the MNIST dataset


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import seaborn
import random
import math


class Neuron:
    # constructor here
    def __init__(self, input_size, my_learning_rate, my_momentum, my_label):
        self.input_size = input_size  # number of inputs - x_i in each example
        self.my_learning_rate = my_learning_rate  # learning rate == step size
        self.my_momentum = my_momentum  # momentum for back-prop
        self.my_label = my_label  # target label
        # create random  initial weight values matrix
        self.weight_vector = np.random.uniform(low=(-.05), high=.05, size=(input_size+1,))  # +1 for bias
        self.delta_weights = np.zeros(self.weight_vector.shape)   # keep track of previous weights for tracking delta

    # train a single example(input_vector)
    def update_weights(self, input_vector, error):
        # add bias for training
        input_vector = np.insert(input_vector, 0, 1, axis=0)
        # update delta_weights taking in error and input with anonymous function
        update_funct = lambda d_w, x_i: (self.my_learning_rate * error * x_i) + (self.my_momentum * d_w)
        # update a single weight
        vector_function = np.vectorize(update_funct)  # can take in lambda or function
        # does the vector function on the entire vector
        self.delta_weights = vector_function(self.delta_weights, input_vector)
        # update weights
        self.weight_vector = self.weight_vector + self.delta_weights

    # return dot product(scalar)
    def get_dot(self, input_vector):
        input_vector = np.insert(input_vector, 0, 1, axis=0)  # add bias
        return np.dot(self.weight_vector, input_vector)  # returns a SCALAR

    # return activation on example
    def get_activation(self, input_vector):
        dot_product = self.get_dot(input_vector)
        activation = (1.0 / (1.0 + (math.exp(-dot_product))))  # sigmoid
        return activation


class TypeClassifier:
    def __init__(self, input_size, num_hidden, num_outputs, c_momentum, c_learning_rate):
        # node = neuron(input_size, my_learning_rate, my_momentum, my_label)
        self.hidden_list = []  # create a list of hidden nodes with their labels
        self.output_list = []  # create a list of output nodes with their labels
        # create hidden nodes labels using ih_ndex
        for h_index in range(num_hidden):    # w_ih = num_inputs+1, num_hidden
            self.hidden_list.append(Neuron(input_size, c_learning_rate, c_momentum, h_index))
        # create output nodes labels using o_index
        for o_index in range(num_outputs):   # w_ho = num_hidden+1, num_outputs
            self.output_list.append(Neuron(num_hidden, c_learning_rate, c_momentum, o_index))

    # train a single example
    def train_classifier_vector(self, input_vector, label): # feed forward
        # FEED FORWARD - calculate activations
        h_activations = [(h_node.get_activation(input_vector)) for h_node in self.hidden_list]
        o_activations = [(o_node.get_activation(h_activations)) for o_node in self.output_list]
        
        # BACK PROPAGATION - calculate error terms
        o_error = []
        h_error = []
        kj_dot = 0

        for k in range(len(o_activations)):
            o_k = o_activations[k]
            t_k = 0.9 if self.output_list[k].my_label == label else 0.1
            o_error.append(o_k*(1-o_k)*(t_k-o_k))

        for k in range(len(o_error)):
            kj_dot += np.dot(self.output_list[k].weight_vector, o_error[k])

        for j in range(len(h_activations)):
            h_j = h_activations[j]
            kj_val = kj_dot[j]
            h_error.append(h_j*(1-h_j)*kj_val)
        
        # BACK PROPAGATION - update weights
        for o in range(len(self.output_list)):
            self.output_list[o].update_weights(h_activations, o_error[o])
        for h in range(len(self.hidden_list)):
            self.hidden_list[h].update_weights(input_vector, h_error[h])

    # train entire dataset
    def train_classifier_dataset(self, input_matrix, label_vector):
        # shuffle list
        zipped_data = list(zip(input_matrix, label_vector))
        np.random.shuffle(zipped_data)
        # pass single example to train
        for input_vector, label in zipped_data:
            self.train_classifier_vector(input_vector, label)

    # return a predicted label
    def classify_vector(self, input_vector): # take in an example and feed it forward to return predicted label
        # feed the example forward
        h_activations = [(h_node.get_activation(input_vector)) for h_node in self.hidden_list]
        o_activations = [(o_node.get_activation(h_activations)) for o_node in self.output_list]
        return np.argmax(o_activations)  # argmax returns position of largest val, which will be == output_node label

    # return output classification list given input array
    def classify_dataset(self, input_matrix):  # classify entire dataset # feed the example forward
        predictions = [(self.classify_vector(input_vector)) for input_vector in input_matrix]
        return predictions

    # return accuracy over dataset
    def get_accuracy(self, input_matrix, labels):
        # get predictions
        predictions = self.classify_dataset(input_matrix)
        comparison_list = list(zip(predictions, labels))
        correct = 0.0
        for prediction, label in comparison_list:
            if int(prediction) == int(label):
                correct += 1.0
        # scale the accuracy data to percent for plotting
        return (correct/len(labels))*100.00


# given a start time, print block time information
def get_time(start_time, end_time, name, optional_num):
    current_time = time.time()
    elapsed = end_time - start_time
    if elapsed == 0:
        elapsed = current_time - start_time
    n_sec = (elapsed % 60)
    n_min = elapsed / 60
    if optional_num < 0:
        print("%s run time: %d minutes, %d seconds" % (name, n_min, n_sec))
    else:
        print("%s %d \nRun time: %d minutes, %d seconds" % (name, optional_num, n_min, n_sec))


# python for main method:
if __name__ == "__main__":

    # Start a timer for program
    start = time.time()

    # Print Header information
    template = "{0:14}{1:20}"
    print("")
    print(template.format("Date:", "04/25/20"))
    print(template.format("Name:", "Sharice Mayer"))
    print(template.format("Topic:", "NN-Multi-Layer-Perceptrons"))
    print("")

    print("----------------------Pre-processing data----------------------")
    print("Experiment 1 - Vary the number of hidden units")
    print("Testing 100 hidden units")
    # print("Experiment 2 - Vary the momentum value")
    # print("Experiment 3 - Vary the number of training examples")

    # Create matrices of data
    train_data = pd.read_csv("data/mnist_train.csv", header=None, sep=',', engine='c', na_filter=False).values
    test_data = pd.read_csv("data/mnist_test.csv", header=None, sep=',', engine='c', na_filter=False).values
    # train_data = pd.read_csv("data/3k_train.csv", header=None, sep=',', engine='c', na_filter=False).values  # 3000
    # test_data = pd.read_csv("data/3k_test.csv", header=None, sep=',', engine='c', na_filter=False).values
    # train_data = pd.read_csv("data/tiny_train.csv", header=None, sep=',', engine='c', na_filter=False).values  # 100
    # test_data = pd.read_csv("data/tiny_test.csv", header=None, sep=',', engine='c', na_filter=False).values

    # create perceptron labels
    train_labels = train_data[:, 0]   # create labels vector for train perceptrons
    test_labels = test_data[:, 0]     # create labels vector for test perceptrons

    # delete the first column (labels) for data computation
    train_data = np.delete(train_data, 0, axis=1)
    test_data = np.delete(test_data, 0, axis=1)

    # pre-process - scale data to 0 to 1 from 0 to 255
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # train perceptrons with 3 learning rates(η this alg) = [0.001, 0.010, 0.100]  # step_size
    plot_num = 1         # plot number for graph figure num
    momentum = 0.9       # η = 0.9 default for back-prop
    learning_rate = 0.1  # Step size = 0.1 default
    num_inputs = 784     # 28x28 pixels per instance
    num_hnodes = 100     # number of hidden nodes
    num_classes = 10     # 0-9 perceptrons (output nodes)
    train_accuracy = []  # for plotting
    test_accuracy = []   # for plotting
    curr_epoch = 0       # track if epochs end early

    # check for even distribution of training data values
    (unique, counts) = np.unique(train_labels, return_counts=True)  # creates two arrays
    frequencies = list(zip(unique, counts))  # zip the elements into tuples
    print("\nChecking training set distribution:")
    print("label\tfrequency")
    for i in frequencies:
        print(i[0], "\t", i[1])
    print("")

    # create a classifier object -  initialise weights and perceptrons
    digit_classifier = TypeClassifier(num_inputs, num_hnodes, num_classes, momentum, learning_rate)

    curr = time.time()
    get_time(start, curr, "Pre-process data", -1)

    print("---------------------------------------------------------------")
    print(f"Epoch {curr_epoch}")
    # calculate accuracy on test and train sets before training:
    train_accuracy.append(digit_classifier.get_accuracy(train_data, train_labels))   # save for confusion matrix
    test_accuracy.append(digit_classifier.get_accuracy(test_data, test_labels))      # save for confusion matrix
    print(f"Training set accuracy: {train_accuracy[curr_epoch]} %")
    print(f"Testing set accuracy: {test_accuracy[curr_epoch]} %")

    get_time(curr, time.time(), "Total current", -1)
    print("---------------------------------------------------------------")
    # print("Running Epochs ... ")
    # print("---------------------------------------------------------------")

    epochs = 50        # 5-10 for pre-testing
    learning = True
    while curr_epoch < epochs and learning is True:
        # update epoch number and timer
        epoch_start = time.time()
        curr_epoch = curr_epoch + 1

        # train perceptrons with training dataset
        digit_classifier.train_classifier_dataset(train_data, train_labels)

        # calculate accuracy after training
        train_accuracy.append(digit_classifier.get_accuracy(train_data, train_labels))
        test_accuracy.append(digit_classifier.get_accuracy(test_data, test_labels))

        # if accuracy isn't changing anymore, no need to continue training
        if abs(train_accuracy[curr_epoch] - train_accuracy[curr_epoch-1]) < 0.001:
            print(f"No change, so end at curr_epoch = {curr_epoch}")
            learning = False

        # brevity option in epoch output for console report img
	# if(curr_epoch == epochs)  # to use brevity option, indent the following print statements

        # print time for single epoch
        epoch_end = time.time()
        get_time(epoch_start, epoch_end, "Epoch", curr_epoch)

        # Print accuracy
        print(f"Training set accuracy: {train_accuracy[curr_epoch]} %")
        print(f"Testing set accuracy: {test_accuracy[curr_epoch]} %")

        # print total running time
        current = time.time()
        get_time(start, current, "Total current", -1)

        print("---------------------------------------------------------------")

    # print final number of epochs trained
    print(f"\nFinal number of epochs = {curr_epoch}")

    # calculate predictions for confusion matrix
    test_predictions = digit_classifier.classify_dataset(test_data)
    # num_examples = len(train_labels)

    # create and save confusion matrix for test data after training
    print(f"Confusion Matrix for learning rate: {learning_rate}:")
    # print(f"test_labels:  {type(test_labels)}  {test_labels.shape}")
    # print(f"test_predictions:  {type(test_predictions)}  {len(test_predictions)}")
    confusion_m = confusion_matrix(test_labels, test_predictions)
    print(confusion_m)
    confusion_df = pd.DataFrame(confusion_m, index=[i for i in "0123456789"], columns=[i for i in "0123456789"])
    plt.figure(figsize=(10, 10))
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    seaborn.heatmap(confusion_df, annot=True, fmt='.1f')
    plt.title('MLNN_Confusion_Matrix_H={0}_M={1}_step={2}_Epochs={3}of{4}'
              .format(num_hnodes, momentum, learning_rate, curr_epoch, epochs))
    plt.savefig(('MLNN_Confusion_H' + str(num_hnodes) + '_M' + str(momentum) + '_Step' + str(learning_rate) + '_Epochs'
                 + str(curr_epoch) + 'of' + str(epochs) + '_Figure_' + str(plot_num) + '.jpg'), bbox_inches='tight')

    # create and save a plot of the data accuracy during training
    plot_num += 1
    plt.figure(figsize=(10, 10))
    plt.axis([0, epochs, 0, 100])  # v = [xmin, xmax, ymin, ymax]
    plt.grid(True)
    x = range(curr_epoch+1)  # whatever the for-loop is+1
    plt.plot(x, train_accuracy, label='Train Accuracy')
    plt.plot(x, test_accuracy, label='Test Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc=2)
    plt.axis('tight')
    plt.title('MLNN_Accuracy_H={0}_M={1}_step={2}_Epochs={3}of{4}'
              .format(num_hnodes, momentum, learning_rate, curr_epoch, epochs))
    plt.savefig(('MLNN_Accuracy_H' + str(num_hnodes) + '_M' + str(momentum) + '_Step' + str(learning_rate) + '_Epochs'
                 + str(curr_epoch) + 'of' + str(epochs) + '_Figure_' + str(plot_num) + '.jpg'), bbox_inches='tight')

    # print total program time
    end = time.time()
    get_time(start, end, "\nTotal program", -1)
    print("")

