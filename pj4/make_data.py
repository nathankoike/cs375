"""
A simple script to make test data for the neural network
"""
import csv, math

# The file containing our data
ifname = "wine.data"
ofname = "wine.csv"

# The output string we will write to a csv
output = ""

# Open the raw data file
with open(ifname) as file:
    # Read the lines of the file
    data = file.readlines()

    # Split the lines of the data
    data = [[eval(term) for term in line.split(',')] for line in data]

# Make a list to hold the mean and SD of the data, disregarding the target
mean_sd_list = [None]

# For every piece of data in a line, disregarding the target (index 0)
for i in range(1, len(data[0])):
    # Compute the mean of the data point
    mean = sum([line[i] for line in data]) / len(data)

    # Compute the standard deviation
    sd = math.sqrt((sum(line[i] - mean for line in data) ** 2) / len(data))

    # Add the mean and sd to the list
    mean_sd_list.append([mean, sd])

### Normalize the data
# For every line in the data
for i in range(len(data)):
    # Make a new line for the data
    new_line = []

    # For everything in the line, disregarding the target
    for j in range(1, len(data[i])):
        # Normalize this piece of data
        new_line.append((data[i][j] - mean_sd_list[j][0]) / mean_sd_list[j][1])

    # Add the target list
    target = [0.0, 0.0, 0.0]
    target[data[i][0] - 1] = 1.0

    # Add the target to the new line
    new_line.append(target)

    # Replace the line in the data
    data[i] = new_line

# Add the headers to the file
for i in range(len(data[0][0:-1])):
    output += f"a{i}" + ','
for i in range(len(data[0][-1])):
    # As long as we don't have the first element
    if i > 0:
        # Add a comma
        output += ','
    output += f"target{i}"

output += '\n'

# Add the data to the file
for line in data:
    # For all the inputs in the line
    for i in range(len(line) - 1):
        # As long as we don't have the first element
        if i > 0:
            # Add a comma
            output += ','
        # Add the thing itself
        output += str(line[i])

    # For all the outputs in the line
    for i in range(len(line[-1])):
        # Add a comma and the output value
        output += ','
        output += str(line[-1][i])

    # Add a newline
    output += '\n'

with open(ofname, 'w') as file:
    file.write(output)
