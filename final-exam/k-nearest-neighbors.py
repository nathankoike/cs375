"""
Name: Nate Koike
"""

import csv, sys, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads data file into a data list and header."""
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
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y = int(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs


def accuracy(training, test, k):
    """Accuracy of k-nearest neighbors alg."""

    true_positives = 0
    total = len(test)

    for (x, y) in test:
        class_prediction = k_nearest_neighbors(training, x, k)
        if class_prediction == y:
            true_positives += 1

    return (true_positives / total)

################################################################################
### k-nearest neighbors

def euclidean(neighbor, query):
    """ Compute the Euclidean distance between the neighbor and the query """
    # Sum the squares of the differences of in every dimension
    squares = sum([(neighbor[i] - query[i]) ** 2 for i in range(len(query))])

    return math.sqrt(squares)


def get_ans_freq(knn):
    """ Get the frequency of the answers in the k nearest neighbors """
    ans_freq = {}

    # Get a count of all the answers
    for i in range(len(knn)):
        # If the answer is already in teh dictionary
        if knn[i][1] in ans_freq:
            ans_freq[knn[i][1]] += 1

        else:
            ans_freq[knn[i][1]] = 1

    return ans_freq


def get_ans(knn):
    """ Get a single answer """
    # Get the frequency of the answers
    ans_freq = get_ans_freq(knn)

    # Find the most common answer and the number of times it occurs
    common = [0, 0]
    for key in ans_freq:
        if ans_freq[key] > common[1]:
            common = [key, ans_freq[key]]

    # Check if there are duplicates of the most common answer
    if list(ans_freq.values()).count(common[1]) > 1:
        # Remove the furthest away neighbor
        knn.pop(knn.index(max(knn)))

        # Recursively try to break the tie
        return get_ans(knn)

    # Return the answer
    return common[0]

def k_nearest_neighbors(training, query, k):
    """Runs k-nearest neighbors algorithm on one test example.
    - training: the labeled training data, which is a list of (x, y) tuples.
                Each x is a list of input floats for the example, and y is an
                integer.
    - query: an input vector x without the known class y.
    - k: the number of neighbors to consider.
    - returns: class prediction, which is an integer."""

    # Create a list of the k nearest neighbrs, filled with placeholder values
    knn = [(9999999999999, None)] * k

    # For every x, y pair in the training data
    for (x, y) in training:
        # Get the Euclicean distance between the data and the query
        distance = euclidean(x, query)

        # If the distance is less than the largest distance in the knn list
        if distance < max(knn)[0]:
            # Replace the maximum distance with the smaller distance and the
            # corresponding answer
            knn[knn.index(max(knn))] = (distance, y)

    # Get a prediction from the knn
    return get_ans(knn)



def main():

    header, data = read_data(sys.argv[1], ",")
    pairs = convert_data_to_pairs(data, header)

    test_header, test_data = read_data(sys.argv[2], ",")
    test_pairs = convert_data_to_pairs(test_data, test_header)

    # print(k_nearest_neighbors(pairs, [3.9, -0.4], 4))
    # print(k_nearest_neighbors(pairs, [3.9, -1.2], 5))

    for k in range(1, 20, 2):
        acc = accuracy(pairs, test_pairs, k)
        print("accuracy({}) = {}".format(k, acc))



if __name__ == "__main__":
    main()
