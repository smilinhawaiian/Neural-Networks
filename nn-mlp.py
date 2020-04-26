# Sharice Mayer
# 04-25-20
# CS445 ML - Spring 2020
# Program #1
# Create a multi-layer-perceptron NN with 1 hidden layer


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
        self.prev_weights = np.zeros(self.weight_vector.shape)   # keep track of previous weights for tracking delta

    # train a single example(input_vector)
    def train(self, input_vector, target):  # target==1 if expected label == perceptron label
        # compute activation value
        activation = 1 if self.test(input_vector) > 0 else 0
        # add bias for training
        input_vector = np.insert(input_vector, 0, 1, axis=0)
        # update weights taking in target and value with inline anonymous function
        update_funct = lambda w, x_in: w + x_in*(self.my_learning_rate * (target - activation))
        # update a single weight
        vector_function = np.vectorize(update_funct)  # can take in lambda or function
        # does the vector function on the entire vector
        self.weight_vector = vector_function(self.weight_vector, input_vector)

    # # test an example
    # def test(self, input_vector):
    #     # insert bias
    #     input_vector = np.insert(input_vector, 0, 1, axis=0)
    #     return np.dot(self.weight_vector, input_vector)  # returns a SCALAR

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
    def train_classifier_vector(self, input_vector, label):
        # check for correct prediction
        prediction = self.classify_vector(input_vector)
        # if prediction incorrect, train perceptron weights for example
        if prediction != label:
            for h_node in self.hidden_list:
                target = 0.9 if label == h_node.my_label else 0.1
                h_node.train(input_vector, target)

    # train entire dataset
    def train_classifier_dataset(self, input_matrix, label_vector):
        # shuffle list
        zipped_data = list(zip(input_matrix, label_vector))
        np.random.shuffle(zipped_data)
        # for each row
        for input_vector, label in zipped_data:
            self.train_classifier_vector(input_vector, label)

    # return a predicted label
    def classify_vector(self, input_vector): # take in an example and feed it forward to return predicted label
        # predictions = [(h_node.test(input_vector)) for h_node in self.hidden_list]
        # return np.argmax(predictions)  # argmax returns position of largest val, which will be == label
        # feed the example forward
        h_activations = [(h_node.get_activation(input_vector)) for h_node in self.hidden_list]
        # h_prediction = np.argmax(h_activations)  # not needed?
        o_activations = [(o_node.get_activation(h_activations)) for o_node in self.output_list]
        return np.argmax(o_activations)

    # return output classification list given input array
    def classify_dataset(self, input_matrix):  # classify entire dataset # feed the example forward
        predictions = [(self.classify_vector(input_vector)) for input_vector in input_matrix]
        # print(f"predictions is {type(predictions)} {len(predictions)}")  # should be 60000?
        return predictions

    # return accuracy over dataset
    def get_accuracy(self, predictions, labels):
        correct = 0
        comparison_list = list(zip(predictions, labels))
        # comparison_list is  <class 'list'> with dimensions 60000
        # print(f"comparison_list is {type(comparison_list)} with dimensions {len(comparison_list)}")
        # print(f"comparison list: {comparison_list}")
        for prediction, label in comparison_list:
            # prediction is <class 'list'> with dimensions 10
            # print(f"prediction is {type(prediction)} with dimensions {len(prediction)}")
            # print(f"label is {type(label)} with dimensions {len(label)}")
            if int(prediction) == int(label):
                correct += 1
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
    print(template.format("Course:", "CS445-ML"))
    print(template.format("Date:", "04/25/20"))
    print(template.format("Name:", "Sharice Mayer"))
    print(template.format("Assignment:", "Program1"))
    print(template.format("Topic:", "NN-Multi-Layer-Perceptrons"))
    print("")

    print("----------------------Pre-processing data----------------------")
    # Create matrices of data
    # train_data = pd.read_csv("data/mnist_train.csv", header=None, sep=',', engine='c', na_filter=False).values
    # test_data = pd.read_csv("data/mnist_test.csv", header=None, sep=',', engine='c', na_filter=False).values
    train_data = pd.read_csv("data/3k_train.csv", header=None, sep=',', engine='c', na_filter=False).values  # 3000
    test_data = pd.read_csv("data/3k_test.csv", header=None, sep=',', engine='c', na_filter=False).values
    # train_data = pd.read_csv("data/tiny_train.csv", header=None, sep=',', engine='c', na_filter=False).values  # 100
    # test_data = pd.read_csv("data/tiny_test.csv", header=None, sep=',', engine='c', na_filter=False).values

    # create perceptron labels
    train_labels = train_data[:, 0]   # create labels vector for train perceptrons
    test_labels = test_data[:, 0]     # create labels vector for test perceptrons
    # # train_labels is <class 'numpy.ndarray'> with dimensions (60000, )
    # print(f"train_labels is {type(train_labels)} with dimensions {train_labels.shape}")
    # # test_labels is <class 'numpy.ndarray'> with dimensions (10000, )
    # print(f"test_labels is {type(test_labels)} with dimensions {test_labels.shape}")

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
    num_hnodes = 25      # number of hidden nodes
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
    train_predictions = digit_classifier.classify_dataset(train_data)
    test_predictions = digit_classifier.classify_dataset(test_data)
    # # train_predictions is <class 'list'> with dimensions 60000
    # print(f"train_predictions is {type(train_predictions)} with dimensions {len(train_predictions)}")
    # # test_predictions is <class 'list'> with dimensions 10000
    # print(f"test_predictions is {type(test_predictions)} with dimensions {len(test_predictions)}")

    train_accuracy.append(digit_classifier.get_accuracy(train_predictions, train_labels))
    test_accuracy.append(digit_classifier.get_accuracy(test_predictions, test_labels))
    print(f"Training set accuracy: {train_accuracy[curr_epoch]} %")
    print(f"Testing set accuracy: {test_accuracy[curr_epoch]} %")

    get_time(curr, time.time(), "Total current", -1)
    print("---------------------------------------------------------------")

    # train for 50 epochs or until accuracy is no longer learning(improving)
    epochs = 10    # 50
    learning = True
    while curr_epoch < epochs and learning is True:
        # update epoch number and timer
        epoch_start = time.time()
        curr_epoch = curr_epoch + 1

        # train perceptrons with training dataset
        # digit_classifier.train_classifier_dataset(train_data, train_labels)

        # calculate accuracy after training
        train_predictions = digit_classifier.classify_dataset(train_data)
        test_predictions = digit_classifier.classify_dataset(test_data)
        train_accuracy.append(digit_classifier.get_accuracy(train_predictions, train_labels))
        test_accuracy.append(digit_classifier.get_accuracy(test_predictions, test_labels))

        # if accuracy isn't changing anymore, no need to continue training
        if abs(train_accuracy[curr_epoch] - train_accuracy[curr_epoch-1]) < 0.001:
            print(f"No change, so end at curr_epoch = {curr_epoch}")
            learning = False

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

    # create and save confusion matrix for test data after training
    print(f"Confusion Matrix for learning rate: {learning_rate}:")
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

    # FOR TESTING
    # train_data = pd.read_csv("data/mnist_train.csv", header=None, sep=',', engine='c', na_filter=False).values
    # test_data = pd.read_csv("data/mnist_test.csv", header=None, sep=',', engine='c', na_filter=False).values
    # train_data = pd.read_csv("data/3k_train.csv", header=None, sep=',', engine='c', na_filter=False).values  # 3000
    # test_data = pd.read_csv("data/3k_test.csv", header=None, sep=',', engine='c', na_filter=False).values
    # train_data = pd.read_csv("data/tiny_train.csv", header=None, sep=',', engine='c', na_filter=False).values  # 100
    # test_data = pd.read_csv("data/tiny_test.csv", header=None, sep=',', engine='c', na_filter=False).values
    # insert bias (input, indice, valueinserted, axis-optional
    # input_vector = np.insert(input_vector, 0, 1, axis=0)  # bias 1 not 1/255
    # plt.savefig((str(epochs) + 'Epoch_Figure' + str(plot_num) + '_Step' + str(learning_rate) + '.jpg'))
    # plt.savefig(('MLP_Figure' + str(plot_num) + '_Step' + str(learning_rate) + '.jpg'), bbox_inches='tight')
    # plt.title('MLNN_Confusion_Matrix_MNIST_training_step={0}_Epochs={1}of{2}'.format(learning_rate, curr_epoch, epochs))
    # plt.savefig(('MLNN_Confusion_Matrix_Step' + str(learning_rate) + '_Epochs' + str(curr_epoch) + 'of' + str(epochs)
    #              + '_Figure_' + str(plot_num) + '.jpg'), bbox_inches='tight')
    # plt.title('MLNN_Accuracy_during_MNIST_training_step={0}_Epochs={1}of{2}'.format(learning_rate, curr_epoch, epochs))
    # plt.savefig(('MLNN_Accuracy_Step' + str(learning_rate) + '_Epochs' + str(curr_epoch) + 'of' + str(epochs)
    #              + '_Figure_' + str(plot_num) + '.jpg'), bbox_inches='tight')
    # plt.title('MLNN_Confusion_Matrix_H={0}_η={1}_step={2}_Epochs={3}of{4}'
    #           .format(num_hnodes, momentum, learning_rate, curr_epoch, epochs))
    # plt.savefig(('MLNN_Confusion_H' + str(num_hnodes) + '_η' + str(momentum) + '_Step' + str(learning_rate) + '_Epochs'
    #              + str(curr_epoch) + 'of' + str(epochs) + '_Figure_' + str(plot_num) + '.jpg'), bbox_inches='tight')
    # plt.title('MLNN_Accuracy_H={0}_η={1}_step={2}_Epochs={3}of{4}'
    #           .format(num_hnodes, momentum, learning_rate, curr_epoch, epochs))
    # plt.savefig(('MLNN_Accuracy_H' + str(num_hnodes) + '_η' + str(momentum) + '_Step' + str(learning_rate) + '_Epochs'
    #              + str(curr_epoch) + 'of' + str(epochs) + '_Figure_' + str(plot_num) + '.jpg'), bbox_inches='tight')

    # CLASSES FOR PRINTING
    # # input_vector is <class 'numpy.ndarray'> with dimensions (784,)
    # print(f"input_vector is {type(input_vector)} with dimensions {input_vector.shape}")
    # # label is <class 'numpy.int64'> with dimensions ()
    # print(f"label is {type(label)} with dimensions {label.shape}")
    # # prediction is <class 'numpy.int64'> with dimensions ()
    # print(f"prediction is {type(prediction)} with dimensions {prediction.shape}")
    # # input_matrix is <class 'numpy.ndarray'> with dimensions (60000, 784)
    # print(f"input_matrix is {type(input_matrix)} with dimensions {input_matrix.shape}")
    # # label_vector is <class 'numpy.ndarray'> with dimensions (60000,)
    # print(f"label_vector is {type(label_vector)} with dimensions {label_vector.shape}")
    # dot_vector is <class 'numpy.float64'> with dimensions ()
    # print(f"dot_vector is {type(dot_vector)} with dimensions {dot_vector.shape}")
    # dot_product is <class 'numpy.float64'> with dimensions ()
    # print(f"dot_product is {type(dot_product)} with dimensions {dot_product.shape}")
    # print(f"prediction = {prediction} label = {label}")  # prediction = 7 label = 2
    # # comparison_list is  <class 'list'> with dimensions 60000
    # print(f"comparison_list is {type(comparison_list)} with dimensions {len(comparison_list)}")
    # print(f"prediction is {type(prediction)} with dimensions {len(prediction)}")
    # print(f"label is {type(label)} with dimensions {len(label)}")

    # MAIN FOR PRINTING
    # # train_data is <class 'numpy.ndarray'> with dimensions (60000, 785)
    # print(f"train_data is {type(train_data)} with dimensions {train_data.shape}")
    # # test_data is <class 'numpy.ndarray'> with dimensions (10000, 785)
    # print(f"test_data is {type(test_data)} with dimensions {test_data.shape}")
    # # train_labels is <class 'numpy.ndarray'> with dimensions (60000, )
    # print(f"train_labels is {type(train_labels)} with dimensions {train_labels.shape}")
    # # test_labels is <class 'numpy.ndarray'> with dimensions (10000, )
    # print(f"test_labels is {type(test_labels)} with dimensions {test_labels.shape}")
    # # train_predictions is <class 'list'> with dimensions 60000
    # print(f"train_predictions: {type(train_predictions)} dim: {len(train_predictions)}")


