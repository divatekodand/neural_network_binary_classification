# MOHAN RAO DIVATE KODANDARAMA
# divatekodand@wisc.edu
# CS USERID: divate-kodanda-rama
#CS760 HW3

# Question
# Homework Assignment #3
# Part A takes 50 points, Part B takes 25 points, and Part C takes 25 points.
# Part A - Programming
# For this part of the assignment, you will be writing code to train and test a neural network with one hidden layer using backpropagation. Specifically, you should assume:
# Your code is intended for binary classification problems.
# All of the attributes are numeric.
# The neural network has connections between input and the hidden layer, and between the hidden and output layer and one bias unit and one output node.
# The number of units in the hidden layer should be equal to the number of input units.
# For training the neural network, use n-fold stratified cross validation.
# Use sigmoid activation function and train using stochastic gradient descent.
# Randomly set initial weights for all units including bias in the range (-1,1).
# Use a threshold value of 0.5. If the sigmoidal output is less than 0.5, take the prediction to be the class listed first in the ARFF file in the class attributes section; else take the prediction to be the class listed second in the ARFF file.
# File Format:
# Your program should read files that are in the ARFF format. In this format, each instance is described on a single line. The feature values are separated by commas, and the last value on each line is the class label of the instance. Each ARFF file starts with a header section describing the features and the class labels. Lines starting with '%' are comments. Your program needs to handle only numeric attributes, and simple ARFF files (i.e. don't worry about sparse ARFF files and instance weights). Your program can assume that the class attribute is named 'class' and it is the last attribute listed in the header section.
#
# Use the following data set for your program : sonar.arff
# Specifications:
# The program should be callable from command line as follows:
# neuralnet trainfile num_folds learning_rate num_epochs
#
# Your program should print the output in the following format for each instance in the source file (in the same order in which the instances appear in the source file)
# fold_of_instance predicted_class actual_class confidence_of_prediction
#
# Here are three example output files, whose file names describe the run parameters.
#
# If you are using a language that is not compiled to machine code (e.g. Java), then you should make a small script called 'neuralnet' that accepts the command-line arguments and invokes the appropriate source-code program and interpreter. More instructions below!
# Part B - Analysis
# In this section, you will draw graphs for analysing the performance of neural network (using sonar.arff as the data set) with respect to certain parameters.
# Plot accuracy of the neural network constructed for 25, 50, 75 and 100 epochs.
# (With learning rate = 0.1 and number of folds = 10)
# Plot accuracy of the neural network constructed with number of folds as 5, 10, 15, 20 and 25.
# (With learning rate = 0.1 and number of epochs = 50)
# Plot ROC curve for the neural network constructed with the following parameters:
# (With learning rate = 0.1, number of epochs = 50, number of folds = 10)
# Please make sure you create three graphs in total (One for each question). Combine all the three graphs in a single PDF file named <wiscid>_analysis.pdf.
# Part C - Written Exercises
# This part consists of some written exercises. Download from here. You can use this latex template to write your solution.
# Submission Instructions
# Create an executable that calls your program as in Homework Assignment #1.
#
# Create a directory named <yourwiscID_hw3> . This directory should contain
# Your source files in a sub-directory named <src>.
# The executable shell script called 'neuralnet'.
# The PDF file '<wiscid>_analysis.pdf' containing the graphs.
# Jar files or any other artifacts necessary to execute your code.
# Compress this directory and submit the compressed zip file in canvas.
# Note:
# You need to ensure that your code will run, when called from the command line as described above, on the CS department Linux machines.
# You WILL be penalized if your program fails to meet any of the above specifications.
# Make sure to test your programs on CSL machines before you submit.


# For Python 2 / 3 compatability
from __future__ import print_function
import sys

from scipy.io import arff
import scipy
from io import StringIO
import numpy as np
import math
# import matplotlib.pyplot as plt
# import sklearn
# from sklearn import metrics

# print("#CS760 HW3")

class Fw_params:

    def __init__(self, x, z1, a1, z2, y_hat):
        self.x = x
        self.z1 = z1
        self.a1 = a1
        self.z2 = z2
        self.y_hat = y_hat

class Model:

    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

class Gradients:

    def __init__(self, dW1, db1, dW2, db2):
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2

def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))

def cross_entropy(yHat, y):
    if y == 1:
      return -np.log2(yHat)
    else:
      return -np.log2(1 - yHat)


def compute_loss(X, Y, model):
    n_examples = X.shape[0]
    loss = 0
    for idx, x in enumerate(X):
        fw_params =  forward(model, x)
        if Y[idx] == 1:
            loss += - np.log2(fw_params.y_hat)
        else:
            loss += - np.log2(1 - fw_params.y_hat)
    loss = (loss * 1.0) / n_examples
    return loss


def forward(model, x):
    W1, b1, W2, b2 = model.W1, model.b1, model.W2, model.b2
    z1 = np.dot(W1, x) + b1
    a1 = scipy.special.expit(z1)
    z2 = np.dot(W2, a1) + b2
    y_hat = scipy.special.expit(z2)
    fw_params = Fw_params(x, z1, a1, z2, y_hat)
    return fw_params

def back_prop(model, x, y, forward_params):
    if y == 1:
        d_y_hat = -1.0 / (forward_params.y_hat + pow(10, -8))
        d_z2 = forward_params.y_hat - 1
    else:
        d_y_hat = 1.0 / (1 - forward_params.y_hat + pow(10, -8))
        d_z2 = forward_params.y_hat

    d_W2 = forward_params.a1 * d_z2
    d_b2 = d_z2
    d_a1 = model.W2 * d_z2
    d_z1 = d_a1 * forward_params.a1 * (1 - forward_params.a1)
    # d_W1 = np.dot(d_z1.reshape((d_z1.shape[0], 1)), x.reshape((1, x.shape[0])))
    d_W1 = np.dot(d_z1.T, x.reshape((1, x.shape[0])))
    d_b1 = d_z1
    d_W2 = d_W2.reshape((1, d_W2.shape[0]))
    d_b1 = d_b1.reshape(d_b1.size)
    grad = Gradients(d_W1, d_b1, d_W2, d_b2)
    return grad

def stratified_sample2(X, Y, num_folds):
    n_examples = X.shape[0]
    n_features = X.shape[1]
    n_ex_fold = math.ceil((n_examples * 1.0) / num_folds)
    if n_examples % num_folds == 0:
        n_ex_last_fold = n_ex_fold
    else:
        n_ex_last_fold = n_examples % n_ex_fold

    pos_list = []
    neg_list = []
    for i in range(n_examples):
        if Y[i] == 0:
            neg_list.append(i)
        else:
            pos_list.append(i)

    n_pos = len(pos_list)
    n_neg = len(neg_list)
    n_pos_fold = math.ceil((n_pos * 1.0) / num_folds)
    n_neg_fold = math.ceil((n_neg * 1.0) / num_folds)

    if n_pos % num_folds == 0:
        n_pos_last_fold = n_pos_fold
    else:
        n_pos_last_fold = n_pos % n_pos_fold

    if n_neg % num_folds == 0:
        n_neg_last_fold = n_neg_fold
    else:
        n_neg_last_fold = n_neg % n_neg_fold

    shuffle_p_list = pos_list
    shuffle_n_list = neg_list
    n_shuffle = 3
    for i in range(n_shuffle):
        np.random.shuffle(shuffle_p_list)
        np.random.shuffle(shuffle_n_list)
    which_fold = {}
    folds = []
    # for fold in range(num_folds - 1):
    #     pos_fold = []
    #     neg_fold = []
    #     for i in range(n_pos_fold):
    #         element = shuffle_p_list.pop()
    #         which_fold[element] = fold
    #         pos_fold.append(element)
    #
    #     for i in range(n_neg_fold):
    #         element = shuffle_n_list.pop()
    #         which_fold[element] = fold
    #         neg_fold.append(element)
    #
    #     folds.append(pos_fold + neg_fold)
    #
    # last_fold = shuffle_p_list + shuffle_n_list
    # for element in last_fold:
    #     which_fold[element] = num_folds - 1
    # folds.append(last_fold)
    for i in range(num_folds):
        folds.append([])

    for i in range(n_pos):
        element = shuffle_p_list.pop()
        which_fold[element] = i % num_folds
        folds[i % num_folds].append(element)

    for i in range(n_neg):
        element = shuffle_n_list.pop()
        which_fold[element] = i % num_folds
        folds[i % num_folds].append(element)

    num_shuffle = 3
    for i in range(num_folds):
        for j in range(num_shuffle):
            np.random.shuffle(folds[i])

    data_folds = []
    label_folds = []
    for fold in folds:
        num_examples = len(fold)
        x_fold = np.zeros((num_examples, n_features), dtype=np.float64)
        y_fold = np.zeros(num_examples, dtype=np.float64)
        for idx in range(num_examples):
            x_fold[idx, :] = X[fold[idx]]
            y_fold[idx] = Y[fold[idx]]
        data_folds.append(x_fold)
        label_folds.append(y_fold)

    return data_folds, label_folds, which_fold

def stratified_sample(X, Y, num_folds):
    n_examples = X.shape[0]
    n_features = X.shape[1]
    n_ex_fold = math.ceil(n_examples / num_folds)
    if n_examples % num_folds == 0:
        n_ex_last_fold = n_ex_fold
    else:
        n_ex_last_fold = n_examples % n_ex_fold

    pos_list = []
    neg_list = []
    for i in range(n_examples):
        if Y[i] == 0:
            neg_list.append(i)
        else:
            pos_list.append(i)

    n_pos = len(pos_list)
    n_neg = len(neg_list)
    n_pos_fold = math.ceil(n_pos / num_folds)
    n_neg_fold = math.ceil(n_neg / num_folds)

    if n_pos % num_folds == 0:
        n_pos_last_fold = n_pos_fold
    else:
        n_pos_last_fold = n_pos % n_pos_fold

    if n_neg % num_folds == 0:
        n_neg_last_fold = n_neg_fold
    else:
        n_neg_last_fold = n_neg % n_neg_fold

    shuffle_p_list = pos_list
    shuffle_n_list = neg_list
    n_shuffle = 3
    for i in range(n_shuffle):
        np.random.shuffle(shuffle_p_list)
        np.random.shuffle(shuffle_n_list)
    which_fold = {}
    folds = []
    for fold in range(num_folds - 1):
        pos_fold = []
        neg_fold = []
        for i in range(n_pos_fold):
            element = shuffle_p_list.pop()
            which_fold[element] = fold
            pos_fold.append(element)

        for i in range(n_neg_fold):
            element = shuffle_n_list.pop()
            which_fold[element] = fold
            neg_fold.append(element)

        folds.append(pos_fold + neg_fold)

    last_fold = shuffle_p_list + shuffle_n_list
    for element in last_fold:
        which_fold[element] = num_folds - 1
    folds.append(last_fold)

    num_shuffle = 3
    for i in range(num_folds):
        for j in range(num_shuffle):
            np.random.shuffle(folds[i])

    data_folds = []
    label_folds = []
    for fold in folds:
        fold.sort()
        num_examples = len(fold)
        x_fold = np.zeros((num_examples, n_features), dtype=np.float64)
        y_fold = np.zeros(num_examples, dtype=np.float64)
        for idx in range(num_examples):
            x_fold[idx, :] = X[fold[idx]]
            y_fold[idx] = Y[fold[idx]]
        data_folds.append(x_fold)
        label_folds.append(y_fold)

    return data_folds, label_folds, which_fold

def train_model(train_data, train_labels, lr, num_epochs):
    input_dim = train_data.shape[1]
    h_dim = input_dim
    out_dim = 1
    W1 = np.random.uniform(low=-1, high=1, size=(h_dim, input_dim))
    b1 = np.random.uniform(low=-1, high=1, size=h_dim)
    W2 = np.random.uniform(low=-1, high=1, size=(1, h_dim))
    b2 = np.random.uniform(low=-1, high=1)

    # W1 = (0.5/h_dim) * np.random.randn(h_dim, input_dim)
    # b1 = (0.5/h_dim) * np.random.randn(h_dim)
    # W2 = (1/h_dim) * np.random.randn(1, h_dim)
    # b2 = (1/h_dim) * np.random.randn()

    model = Model(W1, b1, W2, b2)
    # print("Initial variance : ", np.var(model.W1), np.var(model.b1), np.var(W2), np.var(b2))
    loss = []
    for epoch in range(num_epochs):
        for id,x in enumerate(train_data):
            forward_params = forward(model, x)
            grad = back_prop(model, x, train_labels[id], forward_params)
            model.W1 -= lr * grad.dW1
            model.b1 -= lr * grad.db1
            model.W2 -= lr * grad.dW2
            model.b2 -= lr * grad.db2
        # loss.append(compute_loss(train_data, train_labels, model))
    #plt.plot(loss)
    # print("variance : ", np.var(model.W1), np.var(model.b1), np.var(W2), np.var(b2))
    return model

def get_traindata(folds, folds_y, fold_no):
    #train_data, train_labels =
    n_folds = len(folds)
    first_fold = 1
    for i in range(n_folds):
        if i == fold_no:
            continue
        else:
            if first_fold == 1:
                first_fold = 0
                train_data = folds[i]
                train_labels = folds_y[i]
            else:
                train_data = np.append(train_data, folds[i], axis=0)
                train_labels = np.append(train_labels, folds_y[i])
    return train_data, train_labels


def train_models(X, Y, num_folds, lr, num_epochs):
    #model_list, folds, folds_y, ex_to_fold
    n_examples = X.shape[0]
    models = []
    # TODO: Handling when number of flolds is 1
    if num_folds > 1:
        folds, folds_y, which_fold = stratified_sample2(X, Y, num_folds)
        for idx, fold in enumerate(folds):
            test_data = fold
            test_labels = folds_y[idx]
            train_data, train_labels = get_traindata(folds, folds_y, idx)
            model = train_model(train_data, train_labels, lr, num_epochs)
            models.append(model)
    else:
        if num_folds ==1:
            print("Number of folds is 1. Using whole data to train the model. Using the same again to evaluate the model")
            folds = []
            folds_y = []
            #TODO: Shuffle the data
            folds.append(X)
            folds_y.append(Y)
            train_data, train_labels = folds[0], folds_y[0]
            model = train_model(train_data, train_labels, lr, num_epochs)
            models.append(model)
            which_fold = {}
            for i in range(n_examples):
                which_fold[i] = 0
        else:
            print("Number of folds : %d. Number of Folds should be greater than 1" % num_folds)
            exit(0)

    return models, folds, folds_y, which_fold

def predict_labels(model_list, X, Y, which_fold):
    #confidences, labels = predict_labels(model_list, X, Y, which_fold)
    n_examples = X.shape[0]
    n_features = X.shape[1]
    n_correct = 0
    confidences = np.zeros(n_examples,dtype=np.float64)
    labels = np.zeros(n_examples, dtype=np.int)
    # print(sorted(which_fold.keys()))
    # print(len(sorted(which_fold.keys())))
    for id, x in enumerate(X):
        # print(id, x)
        model_index = which_fold[id]
        fw_params = forward(model_list[model_index], x)
        confidences[id] = fw_params.y_hat
        if confidences[id] < 0.5:
            labels[id] = 0
        else:
            labels[id] = 1
        if labels[id] == Y[id]:
            n_correct += 1
    accuracy = (n_correct * 1.0) / n_examples
    # accuracy = (n_correct) / n_examples
    #print("accuracy :", accuracy)
    return confidences, labels, accuracy



def print_output(X, Y, confidences, labels, which_fold, neg_label, pos_label):
    n_examples = Y.size
    for idx,y in enumerate(Y):
        if labels[idx] == 0:
            predicted = neg_label
        else:
            predicted = pos_label
        if Y[idx] == 0:
            actual = neg_label
        else:
            actual = pos_label
        print(str(which_fold[idx]) + " " + predicted + " " + actual + " " + str(confidences[idx]))

# def partb_01(X, Y):
#     epoch_arr = [25, 50, 75, 100]
#     lr = 0.1
#     n_folds = 10
#     accuracies = []
#     for n_epoch in epoch_arr:
#         model_list, folds, folds_y, which_fold = train_models(X, Y, n_folds, lr, n_epoch)
#         confidences, labels, accuracy= predict_labels(model_list, X, Y, which_fold)
#         accuracies.append(accuracy)
#
#     for i in range(len(accuracies)):
#         accuracies[i] = accuracies[i] * 100.0
#
#     plt.plot(epoch_arr,accuracies)
#     plt.xlim(10, 110)
#     plt.ylim(50, 95)
#     plt.xlabel('Number of epochs')
#     plt.ylabel('Accuracy')
#     plt.title('PARTB 01 - Plot showing the variation of Accuracy with number of Epochs')
#     plt.show()
#
# def partb_02(X, Y):
#     n_folds_arr = [5,10,15,20,25]
#     lr = 0.1
#     n_epoch = 50
#     accuracies = []
#     for n_folds in n_folds_arr:
#         model_list, folds, folds_y, which_fold = train_models(X, Y, n_folds, lr, n_epoch)
#         confidences, labels, accuracy= predict_labels(model_list, X, Y, which_fold)
#         accuracies.append(accuracy)
#
#     for i in range(len(accuracies)):
#         accuracies[i] = accuracies[i] * 100.0
#
#     plt.plot(n_folds_arr,accuracies)
#     plt.xlim(0, 30)
#     plt.ylim(50, 95)
#     plt.xlabel('Number of folds')
#     plt.ylabel('Accuracy')
#     plt.title('PARTB 02 - Plot showing the variation of Accuracy with number of Folds')
#     plt.show()
#     print("part 2")
#
# # Plot ROC curve for the neural network constructed with the following parameters:
# # (With learning rate = 0.1, number of epochs = 50, number of folds = 10)
# def partb_03(X, Y):
#     lr = 0.1
#     n_epoch = 50
#     n_folds = 10
#     model_list, folds, folds_y, which_fold = train_models(X, Y, n_folds, lr, n_epoch)
#     confidences, labels, accuracy = predict_labels(model_list, X, Y, which_fold)
#     print(confidences)
#     print(labels)
#     fpr, tpr, thresholds = metrics.roc_curve(Y, confidences, pos_label=1)
#     print(fpr)
#     print(tpr)
#     print(thresholds)
#
#     plt.figure()
#     lw = 2
#     plt.plot(fpr, tpr, color='darkorange', lw=lw, label=('ROC curve - learning rate = %f, epochs = %d, folds = %d' %(lr, n_epoch, n_folds)))
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('PARTB 03 - Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()



def main():
    '''
    Loads the data
    '''

    ##PROCESS THE ARGURMENT
    # print(sys.argv)
    num_args = len(sys.argv)
    # How to call the program from commandline - neuralnet trainfile num_folds learning_rate num_epochs
    # example - neuralnet sonar.arff 10 0.5 10
    if (num_args < 5):
        print("Wrong Usage - Script takes 4 arguments")
        print("Example Usage- python neuralnet.py sonar.arff 10 0.5 10")
        exit(0)
    train_filename = sys.argv[1]
    num_folds = int(sys.argv[2])
    lr = float(sys.argv[3])
    num_epochs = int(sys.argv[4])
    # print(train_filename, num_folds, lr, num_epochs)

    ##LOAD THE DATA
    train_file = open(train_filename, 'r')
    train_set, train_meta = arff.loadarff(train_file)
    num_examples = len(train_set)
    num_features = len(train_set[1]) - 1
    # print(type(train_set[1]))
    # print(train_set.size)

    train_attributes = train_meta.names()  # Returns a list
    train_attribute_types = train_meta.types()  # Returns a list
    # print(train_attribute_types)
    output_labels = train_meta.__getitem__(train_attributes[-1])
    neg_label = output_labels[-1][0]
    pos_label = output_labels[-1][1]
    # print(neg_label, pos_label)

    numeric_to_labels = {}
    labels_to_numeric = {}
    for i in range(2):
        numeric_to_labels[i] = output_labels[-1][i]
        labels_to_numeric[output_labels[-1][i]] = i

    X_list = []
    y_list = []
    for example in train_set:
        example_list = list(example)
        example_x = example_list[:-1]
        X_list.append(example_x)
        if example_list[-1].decode('UTF-8') == neg_label:
            y_list.append(0)
        else:
            y_list.append(1)
    # print(X_list, len(X_list))
    # print(y_list, len(y_list))

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(y_list, dtype=np.int)
    # print(X, type(X))
    # print(y, type(y))

    #PART B
    # partb_01(X,Y)
    # partb_02(X,Y)
    # partb_03(X,Y)
    # PART A
    model_list, folds, folds_y, which_fold = train_models(X, Y, num_folds, lr, num_epochs)
    confidences, labels, accuracy= predict_labels(model_list, X, Y, which_fold)
    # print(accuracy)
    print_output(X, Y, confidences, labels, which_fold, neg_label, pos_label)


if __name__ == "__main__":
    main()