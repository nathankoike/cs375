"""
linear_classifier.py
Usage: python3 linear_classifier.py data_file.csv
"""

import csv, sys, random, math

ALPHA = 0.5

################################################################################
### Utility functions

def read_data(filename, delimiter=",", has_header=True):
    """Reads data from filename. The optional parameters can allow it
    to read data in different formats. Returns a list of headers and a
    list of lists of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        pair = (example[:-1], example[-1])
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def weights_to_slope_intercept(weights):
    """Turns weights into slope-intercept form for 2D spaces"""

    slope = - weights[1] / weights[2]
    intercept = - weights[0] / weights[2]

    return (slope, intercept)

################################################################################
### Perceptron: Linear Classifier with a Hard Threshold

def perceptron(training, epochs):
    """We will implement a perceptron, i.e. linear classification with a
    hard threshold."""

    weights = [random.random() for _ in range(len(training[0][0]))]

    # Stochastic gradient descent:
    for e in range(epochs):
        # Choose a random training example
        example = random.choice(training)
        old_weights = weights[:]

        alpha = 1000 / (1000 + e)
        perceptron_learning_rule(alpha, weights, example)

        if weights != old_weights:
            print()
            print("Epoch:", e)
            (slope, intercept) = weights_to_slope_intercept(weights)

            print("y = {} x + {}".format(slope, intercept))
            print("accuracy:", accuracy_for_perceptron(weights, training))

    return weights

def accuracy_for_perceptron(weights, pairs):
    """Calculates the accuracy as the proportion of examples that these
    weights correctly classify."""

    correct = 0
    total = len(pairs)

    for (x, y) in pairs:
        if perceptron_hypothesis(weights, x) == y:
            correct += 1

    return correct / total


def perceptron_learning_rule(alpha, weights, example):
    """Learning rule for perceptron/hard threshold."""

    (x, y) = example

    hyp = perceptron_hypothesis(weights, x)

    if y != hyp:
        for i in range(len(weights)):
            weights[i] = weights[i] + (alpha * (y - hyp) * x[i])

def perceptron_hypothesis(weights, x):
    """Implements the perceptron classification hypothesis."""

    if dot_product(weights, x) >= 0:
        return 1
    return 0


###########################
## Logistic classification

def logistic_classification(training, epochs):
    """We will implement logist classification"""

    weights = [random.random() for _ in range(len(training[0][0]))]

    # Stochastic gradient descent:
    for e in range(epochs):
        # Choose a random training example
        example = random.choice(training)
        old_weights = weights[:]

        alpha = 1000 / (1000 + e)

        logistic_learning_rule(alpha, weights, example)

        if weights != old_weights:
            print()
            print("Epoch:", e)
            (slope, intercept) = weights_to_slope_intercept(weights)

            print("y = {} x + {}".format(slope, intercept))
            print("accuracy:", accuracy_for_logistic_classification(weights, training))

    return weights

def logistic_learning_rule(alpha, weights, example):
    """Learning rule for logistic."""

    (x, y) = example

    hyp = logistic_hypothesis(weights, x)

    for i in range(len(weights)):
        weights[i] = weights[i] + (alpha * (y - hyp) * hyp * (1 - hyp) * x[i])

def logistic(x):
    """Logistic / signmoid function."""

    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0

    return 1.0 / denom

def logistic_hypothesis(weights, x):
    """Hypothesis for logistic classification."""

    return logistic(dot_product(weights, x))

def accuracy_for_logistic_classification(weights, examples):
    """Finds the accuracy of these weights on these examples.
    Uses mean squared error."""

    error = 0
    total = len(examples)

    # Calculate the squared error for each example:
    for (x, y) in examples:
        error += (logistic_hypothesis(weights, x) - y) ** 2

    return 1 - (error / total)

def main():

    # # Read data from the file provided at command line
    # header, data = read_data(sys.argv[1], ",")
    #
    # # Convert data into (x, y) tuples
    # example_pairs = convert_data_to_pairs(data)
    #
    # # Insert 1.0 as first element of each x to work with the dummy weight
    # training = [([1.0] + x, y) for (x, y) in example_pairs]
    #
    # # See what the data looks like
    # # for (x, y) in training:
    # #     print("x = {}, y = {}".format(x, y))
    #
    # # Run linear classification. This is what you need to implement
    # w = perceptron(training, 10000)
    #
    # # Read data from the file provided at command line
    # header, data_test = read_data(sys.argv[2], ",")
    #
    # # Convert data into (x, y) tuples
    # example_pairs_test = convert_data_to_pairs(data_test)
    #
    # test = [([1.0] + x, y) for (x, y) in example_pairs_test]
    #
    # # Test our trained model on unseen test data
    # accuracy_on_unseen_data = accuracy_for_perceptron(w, test)
    #
    # print("Accuracy on unseen data:", accuracy_on_unseen_data)


    # Read data from the file provided at command line
    header, data = read_data(sys.argv[1], ",")

    # Convert data into (x, y) tuples
    example_pairs = convert_data_to_pairs(data)

    # Insert 1.0 as first element of each x to work with the dummy weight
    training = [([1.0] + x, y) for (x, y) in example_pairs]

    # See what the data looks like
    # for (x, y) in training:
    #     print("x = {}, y = {}".format(x, y))

    # Run logistic classification. This is what you need to implement
    w = logistic_classification(training, 1000)

    # Read data from the file provided at command line
    header, data_test = read_data(sys.argv[2], ",")

    # Convert data into (x, y) tuples
    example_pairs_test = convert_data_to_pairs(data_test)

    test = [([1.0] + x, y) for (x, y) in example_pairs_test]

    # Test our trained model on unseen test data
    accuracy_on_unseen_data = accuracy_for_logistic_classification(w, test)

    print("Accuracy on unseen data:", accuracy_on_unseen_data)


if __name__ == "__main__":
    main()
