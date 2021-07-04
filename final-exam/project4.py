"""
This neural network is implemented in a with OOP, and as such has a high degree
of flexibility. As its input, it takes raw or normalized data, without the
leading 1 value needed for a neural network implemented with an adjacency
matrix. Create a new neural net with the following syntax:

myNeuralNetwork = NeuralNetwork([layerSizes], epochs=1000)

The number of epochs is an optional parameter that defaults to 1000. Train the
neural network by providing a list of data in the pattern (x, y), and optionally
a True/False value to toggle stohcastic gradient descent. k-fold cross
validation works in much the same way, but with an optional k parameter that
defaults to 5. Run these commands with the following syntax:

myNeuralNetwork.back_propagation_learning([data], stochastic=False)
myNeuralNetwork.cross_validation_back_prop([data], k=5, stochastic=False, verbose=False)

The optional verbose parameter toggles verbosity in the accuracy testing.

Finally, to clear the trained weights in the neural net, as is needed during
cross validation, use the following syntax:

myNeuralNetwork.reset()


Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math

ALPHA = 0.5

################################################################################
### Utility functions

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
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


def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs


def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum


def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom


def accuracy(nn, pairs, verbose=False):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        # nn.forward_propagate(x)
        class_prediction = nn.predict_class(x)
        # class_prediction = nn.predict(x)
        if class_prediction == y:
            true_positives += 1
        elif verbose:
            print(y, nn.predict(x))

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return (true_positives / total)

################################################################################
## Logistic classification

def sigmoid(x):
    """Logistic / signmoid function."""

    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0

    return 1.0 / denom

################################################################################
### Neural Network code goes here

class Node:
    """ A single node in a layer in a neural network """
    def __init__(self, inputs):
        """ inputs: the number of inputs to the node """
        # The bias of this node
        self.bias = ALPHA

        # The last input the node has seen
        self.input = None

        # The activation of this node (once activated)
        self.activation = None

        # All the weights on all the inputs
        self.weights = []

        # The error of this node (calculated during backpropagation)
        self.delta = None

        # The error of each weight
        self.weights_delta = []

        # Initialize all the weights to some random value between -1 and 1
        for _ in range(inputs):
            # Set a random weight for the input
            self.weights.append(random.uniform(-1, 1))

            # Set the error on that weight to be 0
            self.weights_delta.append(0)


    def reset(self):
        """ Reset all the trained data in this node """
        # For every weight
        for i in range(len(self.weights)):
            # Reset the weights
            self.weights[i] = random.uniform(-1, 1)

            # Reset the weights delta
            self.weights_delta[i] = 0

        # Reset the delta of this node
        self.delta = None

        # Reset the activation of this node
        self.activation = None

        # Reset the bias of this node
        self.bias = ALPHA

        # Reset the last input the node has seen
        self.input = None


    def activate(self, inputs):
        """ Calculate the activation of one node based on a list of numbers as
            the input """
        # Save the inputs
        self.input = inputs

        # The sigmoid function of the dot product of the weights and the inputs
        self.activation = sigmoid(dot_product(self.weights, inputs) + self.bias)

        return self.activation


class Layer:
    """ A single layer in a neural network """
    def __init__(self, nodes, inputs):
        """
        nodes: the number of nodes in this layer
        inputs: the number of inputs to this layer
        input_layer: is this the input layer? """

        # A list of all the nodes in this layer
        self.nodes = []

        # Initialize all the nodes with random weights
        for _ in range(nodes):
            self.nodes.append(Node(inputs))

        self.activation = None


    def reset(self):
        """ Reset all the trained data in this layer """
        for node in self.nodes:
            node.reset()


    def activate(self, inputs):
        """ Activate an entire layer in the neural net """

        # Activate every node in this layer
        for node in self.nodes:
            node.activate(inputs)

        # Save the activation
        self.activation = [node.activation for node in self.nodes]

        return self.activation


    def back_propagate(self, next):
        """ Perform back propagation on this layer """
        # For every node in this layer
        for i in range(len(self.nodes)):
            # Track the delta across this node
            node_delta = sum([n.delta * n.weights[i] for n in next.nodes])

            # Set the error of this node
            self.nodes[i].delta = node_delta

        # Return this layer to be used again
        return self


    def update_weights(self):
        """ Update the weights in this layer """
        # For every node in the layer
        for node in self.nodes:
            # For every weight in the weights of the node
            for i in range(len(node.weights)):
                # Add the product of previous error on the weight and the ALPHA
                # value to the weight
                node.weights[i] += ALPHA * node.weights_delta[i]

                # Update the error on the weight
                node.weights_delta[i] = node.delta * node.input[i]

                # Add the updated error on the weight to the weight
                node.weights[i] += node.weights_delta[i]

            # Update the bias in the node
            node.bias += ALPHA * node.weights_delta[len(node.weights) - 1]
            node.bias += node.delta


class NeuralNetwork:
    """ A neural network with any number of nodes per layer """
    def __init__(self, layers, epochs=1000):
        """ layers: a list of the number of nodes in each layer """

        # The number of epochs for which the neural net will train
        self.epochs = epochs

        # All the layers in the neural network
        self.layers = []

        # The first layer is the input layer, which needs no weights
        for i in range(len(layers) - 1):
            # Initialize all the layers randomly
            self.layers.append(Layer(layers[i + 1], layers[i]))


    def reset(self):
        """ Reset all the trained weights and biases """
        for layer in self.layers:
            layer.reset()


    def forward_propagate(self, inputs):
        """ forward propagate without returning anything """
        # Copy the inputs
        results = inputs[:]

        # For every layer in the neural net
        for layer in self.layers:
            # Use the results from the previous layer as the input of the next
            results = layer.activate(results)


    def predict(self, inputs):
        """ Forward propagation to get a prediction from the neural net """
        # Copy the inputs
        results = inputs[:]

        # For every layer in the neural net
        for layer in self.layers:
            # Use the results from the previous layer as the input of the next
            results = layer.activate(results)

        return results


    def predict_class(self, inputs=None):
        """ Predict the output and round the answer to the nearest integer """
        # If we have inputs
        if inputs:
            return [float(round(ans)) for ans in self.predict(inputs)]

        return [float(round(ans)) for ans in self.layers[-1].activation[:]]


    def back_propagate(self, expected, output):
        """ Back propagate the result to set error values """
        # For each node in the output layer
        for i in range(len(self.layers[-1].nodes)):
            node = self.layers[-1].nodes[i]

            # Find some error with the ith output
            node.delta = output[i] * (1 - output[i]) * (expected[i] - output[i])

        # Save this layer for back propagation
        forward_layer = self.layers[-1]

        # For every hidden layer, working backwards
        for layer in self.layers[:-1][::-1]:
            # Call back propagation on the previous layer
            forward_layer = layer.back_propagate(forward_layer)


    def back_propagation_learning(self, data, stochastic=False):
        """ train the weights in the layers of the neural network based on the
            error in the data """
        # Loop for some number of epochs (longer is generally better)
        for _ in range(self.epochs):
            # For each example (x, y, ...) in examples or data
            # Use a random example for stochastic gradient descent
            if stochastic:
                [x, y] = data[random.randint(0, len(data) - 1)]

                # Make a prediction with forward propagation and get the results
                results = self.predict(x)

                # Back propagate the results
                self.back_propagate(y, results)

                for layer in self.layers:
                    # Update the weights in the layer
                    layer.update_weights()

            else:
                # Use every example for gradient descent
                for example in data:
                    # Get the inputs (x) and their proper outputs (y)
                    [x, y] = example

                    # Make a prediction with forward propagation and get the results
                    results = self.predict(x)

                    # Back propagate the results
                    self.back_propagate(y, results)

                    for layer in self.layers:
                        # Update the weights in the layer
                        layer.update_weights()


    def cross_validation_back_prop(self, data, k=5, stochastic=False, verbose=False):
        """ Perform k-fold cross validation on the neural net to check if the
            architecture is good """
        # Copy the data
        new_data = data[:]

        # A list to hold all the groups
        groups = []

        # Split the data into k groups
        for f in range(k - 1):
            group = []

            # Get an even distribution of data across all groups
            for _ in range(len(data) // k):
                # Remove a random element from the data and add it to the
                # current group
                group.append(new_data.pop(random.randint(0, len(new_data) - 1)))

            # Add the new group to the list of groups
            groups.append(group)

        # Add the last group
        groups.append(new_data)

        # The accuracies of the training sets
        accuracies = []

        # Run k sets of training on the data, leaving one set for validation
        for i in range(k):
            # Reset all the trained data
            self.reset()

            # Hold the new training data
            training_data = []

            # Make the training data one list
            for j in range(len(groups)):
                # If the index isn't the one we're avoiding
                if i != j:
                    training_data += groups[j]

            # Run the training
            self.back_propagation_learning(training_data, stochastic)

            # Validate the results
            accuracies.append(accuracy(self, groups[i], verbose))

        # Return the average of the accuracies
        return sum(accuracies) / len(accuracies)


def main():
    header, data = read_data(sys.argv[1], ",")
    header2, data2 = read_data(sys.argv[2], ",")

    training = convert_data_to_pairs(data, header)
    testing = convert_data_to_pairs(data2, header2)

    ## I expect the running of your program will work something like this;
    ## this is not mandatory and you could have something else below entirely.
    nn = NeuralNetwork([2, 8, 5], 500)

    # print("cross validated accuracy:", nn.cross_validation_back_prop(pairs))

    # nn.reset()

    nn.back_propagation_learning(training)
    print("regular training accuracy:", accuracy(nn, testing))




if __name__ == "__main__":
    main()
