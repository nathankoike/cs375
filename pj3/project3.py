"""
Author:         Nate Koike & Zhaosen Guo
Date:           2021/4/15
Description:    Implement Genetic Programming fuction that tries to solve
                the given problem.
"""

import operator, random, math, csv, copy

MAX_FLOAT = 1e12
OUTPUT = "new_results.txt"

def safe_division(numerator, denominator):
    """Divides numerator by denominator. If denominator is close to 0, returns
    MAX_FLOAT as an approximate of infinity."""
    if abs(denominator) <= 1 / MAX_FLOAT:
        return MAX_FLOAT
    return numerator / denominator

def safe_exp(power):
    """Takes e^power. If this results in a math overflow, or is greater
    than MAX_FLOAT, instead returns MAX_FLOAT"""
    try:
        result = math.exp(power)
        if result > MAX_FLOAT:
            return MAX_FLOAT
        return result
    except OverflowError:
        return MAX_FLOAT

def safe_log(arg):
    """ Take the natural log of the argument """
    try:
        if arg <= 0:
            return 0
        result = math.log(arg)
        if result < -MAX_FLOAT:
            return -MAX_FLOAT
        return result
    except OverflowError:
        return -MAX_FLOAT

# Dictionary mapping function stings to functions that perform them
FUNCTION_DICT = {"+": operator.add,
                 "-": operator.sub,
                 "*": operator.mul,
                 "/": safe_division,
                 "exp": safe_exp,
                 "sin": math.sin,
                 "cos": math.cos,
                 "log": safe_log}

# Dictionary mapping function strings to their arities (number of arguments)
FUNCTION_ARITIES = {"+": 2,
                    "-": 2,
                    "*": 2,
                    "/": 2,
                    "exp": 1,
                    "sin": 1,
                    "cos": 1,
                    "log": 1}

# List of function strings
FUNCTIONS = list(FUNCTION_DICT.keys())

# Strings for each variable
VARIABLES = ["x0", "x1"]


class TerminalNode:
    """Leaf nodes that contain terminals."""

    def __init__(self, value):
        """value might be a literal (i.e. 5.32), or a variable as a string."""
        self.value = value

    def __str__(self):
        return str(self.value)

    def eval(self, variable_assignments):
        """Evaluates node given a dictionary of variable assignments."""

        if self.value in VARIABLES:
            return variable_assignments[self.value]

        return self.value

    def tree_depth(self):
        """Returns the total depth of tree rooted at this node.
        Since this is a terminal node, this is just 0."""

        return 0

    def size_of_subtree(self):
        """Gives the size of the subtree of this node, in number of nodes.
        Since this is a terminal node, this is just 1."""

        return 1

    def get_subtrees_range(self, size):
        """ Return a list of subtrees with appropriate size. """
        # A list to hold all the target subtrees
        trees = []

        # If we are below the lower threshold of the acceptable size range
        if self.size_of_subtree() < size // 2:
            return [] # Return nothing

        return [self]


class FunctionNode:
    """Internal nodes that contain functions."""

    def __init__(self, function_symbol, children):
        self.function_symbol = function_symbol
        self.function = FUNCTION_DICT[self.function_symbol]
        self.children = children

        assert len(self.children) == FUNCTION_ARITIES[self.function_symbol]

    def __str__(self):
        """This should make printed programs look like Lisp."""

        result = f"({self.function_symbol}"
        for child in self.children:
            result += " " + str(child)
        result += ")"
        return result

    def eval(self, variable_assignments):
        """Evaluates node given a dictionary of variable assignments."""

        # Calculate the values of children nodes
        children_results = [child.eval(variable_assignments) for child in self.children]

        # Apply function to children_results.
        return self.function(*children_results)

    def tree_depth(self):
        """Returns the total depth of tree rooted at this node."""

        return 1 + max(child.tree_depth() for child in self.children)

    def size_of_subtree(self):
        """Gives the size of the subtree of this node, in number of nodes."""
        return 1 + sum(child.size_of_subtree() for child in self.children)

    def get_subtrees_range(self, size):
        """ Return a list of subtrees with appropriate size. """
        # A list to hold all the target subtrees
        trees = []

        # If we are below the lower threshold of the acceptable size range
        if self.size_of_subtree() < size // 2:
            return [] # Return nothing

        # Check if we can add this subtree
        if self.size_of_subtree() <= size + size // 2:
            trees.append(self)

        # Recursively search the entire tree
        for child in self.children:
            trees += child.get_subtrees_range(size)

        # Return all the appropriately sized trees
        return trees


class Individual:
    """ An individual program to be used with GP """

    def __init__(self, syntax_tree):
        self.tree = syntax_tree

        # We will deal with these later in testing
        self.error_vector = []
        self.total_error = 0

    def __str__(self):
        """ Nicely provide all the relevant information about this program """
        result = "Syntax Tree:\n\t" + str(self.tree)
        result += "\nError Vector:\n\t" + str(self.error_vector)
        result += "\nTotal Error: " + str(self.total_error)
        return result

    def eval(self, variable_assignments):
        """ Evaluate the program """
        return self.tree.eval(variable_assignments)

    def test(self, test_cases):
        """ Run all the program on all the provided tets cases and save all the
            error vectors and total errors """
        # Make sure the error vector and total error are zeroed out
        self.error_vector = []
        self.total_error = 0

        # All the results of the program on every test case
        results = []

        # For every test case
        for variable_assignments in test_cases:
            # Run the program
            result = self.tree.eval(variable_assignments)
            results.append(result)

            # Add the error value to the error vector
            self.error_vector.append(variable_assignments['y'] - result)

        # Get the total error
        self.total_error = sum(abs(error) for error in self.error_vector)

        return results


# ==============================================================================
# =========================== Program Generation ===============================
# ==============================================================================


def random_terminal():
    """Returns a random TerminalNode
    Half the time pick a random variable, and half the time a random float in
    the range [-10.0, 10.0]"""

    if random.random() < 0.5:
        terminal_value = random.choice(VARIABLES)
    else:
        terminal_value = random.uniform(-10.0, 10.0)

    return TerminalNode(terminal_value)


def generate_tree_full(max_depth):
    """Generates and returns a new tree using the Full method for tree
    generation and a given max_depth."""

    if max_depth <= 0:
        return random_terminal()

    function_symbol = random.choice(FUNCTIONS)
    arity = FUNCTION_ARITIES[function_symbol]
    children = [generate_tree_full(max_depth - 1) for _ in range(arity)]

    return FunctionNode(function_symbol, children)


def generate_tree_grow(max_depth):
    """Generates and returns a new tree using the Grow method for tree
    generation and a given max_depth."""

    ## What percent of the time do we want to select a terminal?
    percent_terminal = 0.25

    if max_depth <= 0 or random.random() < percent_terminal:
        return random_terminal()

    function_symbol = random.choice(FUNCTIONS)
    arity = FUNCTION_ARITIES[function_symbol]
    children = [generate_tree_grow(max_depth - 1) for _ in range(arity)]

    return FunctionNode(function_symbol, children)


def generate_random_program(size_range=[1, 3]):
    """Creates a random program as a syntax tree.
    This uses Ramped Half-and-Half.
    max-depth taken from the range [2, 5] inclusive."""

    depth = random.randint(size_range[0], size_range[1])
    if random.random() < 0.5:
        return generate_tree_full(depth)
    else:
        return generate_tree_grow(depth)

# ==============================================================================
# ================= Parent Selection and Variation Operators ===================
# ==============================================================================

def subtree_at_index(node, index):
    """Returns subtree at particular index in this tree. Traverses tree in
    depth-first order."""
    if index == 0:
        return node

    # Subtract 1 for the current node
    index -= 1

    # Go through each child of the node, and find the one that contains this index
    for child in node.children:
        child_size = child.size_of_subtree()

        if index < child_size:
            return subtree_at_index(child, index)

        index -= child_size

    return "INDEX {} OUT OF BOUNDS".format(index)


def replace_subtree_at_index(node, index, new_subtree):
    """Replaces subtree at particular index in this tree. Traverses tree in
    depth-first order."""
    # Return the subtree if we've found index == 0
    if index == 0:
        return new_subtree

    # Subtract 1 for the current node
    index -= 1

    # Go through each child of the node, and find the one that contains this index
    for child_index in range(len(node.children)):
        child_size = node.children[child_index].size_of_subtree()

        if index < child_size:
            new_child = replace_subtree_at_index(node.children[child_index], index, new_subtree)
            node.children[child_index] = new_child
            return node

        index -= child_size

    return "INDEX {} OUT OF BOUNDS".format(index)


def random_subtree(program):
    """Returns a random subtree from given program, selected uniformly."""
    nodes = program.size_of_subtree()
    node_index = random.randint(0, nodes - 1)

    return subtree_at_index(program, node_index)


def replace_random_subtree(program, new_subtree):
    """Replaces a random subtree with new_subtree in program, with node to
    be replaced selected uniformly."""
    nodes = program.size_of_subtree()
    node_index = random.randint(0, nodes - 1)
    new_program = copy.deepcopy(program)

    return replace_subtree_at_index(new_program, node_index, new_subtree)


def tournament_selection(programs):
    """ Pick the parent with the lowest error and return it along with the list
        of programs without that program included """
    # Set a really high best error and an impossible index
    best_error = MAX_FLOAT
    best_index = -1

    # For every program in the list
    for i in range(len(programs)):
        # If we found a better error
        if programs[i].total_error < best_error:
            # Save the best program and the best index
            best_error = programs[i].total_error
            best_index = i

    return programs[best_index]


def crossover(parent_a, parent_b):
    """ Perform a normal crossover operation and return a new individual with
        resulting from the crossover """
    # Select a random subtree from Parent B
    replacement = random_subtree(parent_b.tree)

    # Randomly insert the subtree from Parent B into the tree from Parent A
    new_tree = replace_random_subtree(parent_a.tree, replacement)

    # Make a new individual with the new tree and return it
    return Individual(new_tree)


def subtree_mutation(parent):
    """ Perform standard subtree mutation and return a new individual with the
        resulting mutation """
    # Generate a new random subtree/program
    new_tree = generate_random_program()

    # Replace a random subtree in the parent's tree with the new tree
    return Individual(replace_random_subtree(parent.tree, new_tree))


def hoist_mutation(parent):
    """ Return an individual whose tree is just a subtree from the parent,
        chosen randomly. """

    return Individual(random_subtree(parent.tree))


def get_size_range(tree):
    """ Return a list with the upper and lower bounds of a size-fair tree """
    lower_bound = tree.size_of_subtree() // 2

    return [lower_bound, tree.size_of_subtree() + lower_bound]


def within_acceptable_range(target, attempt):
    """ Determine whether or not the attmept is with the acceptable size
        range of the target """
    [lower_bound, upper_bound] = get_size_range(target)

    # Check the lower bound
    if attempt.size_of_subtree() < lower_bound:
        return False

    # Check the upper bound
    if attempt.size_of_subtree() > upper_bound:
        return False

    return True


def sf_crossover(parent_a, parent_b, d_limit, s_limit):
    """ Attempt to perform crossover while maintaining some semblance of a size
        and depth limit. """
    # Get a random index in the range of the length of Parent A
    nodes = parent_a.tree.size_of_subtree()
    index = random.randint(0, nodes - 1)

    # Get the subtree at the index in Parent A to replace
    to_replace = subtree_at_index(parent_a.tree, index)

    # If the size range of the trees is incompatible, perform normal crossover
    if not within_acceptable_range(to_replace, parent_b.tree):
        return crossover(parent_a, parent_b)

    # Make a list of all acceptable subtrees in Parent B
    size = to_replace.size_of_subtree()
    appropriate_trees = parent_b.tree.get_subtrees_range(size)

    # If there are no appropriate subtrees in Parent B
    if appropriate_trees == []:
        # Perform a hoist mutation on both parents and crossover on the results
        return crossover(hoist_mutation(parent_a), hoist_mutation(parent_b))

    # Pick a random subtree from the list and replace the subtree in Parent A
    # with the subtree from the list
    new_tree = replace_subtree_at_index(
        copy.deepcopy(parent_a.tree),
        index,
        random.choice(appropriate_trees))

    # Create an individual with the new tree
    program = Individual(new_tree)

    # Enforce the size limit
    if new_tree.size_of_subtree() >= s_limit:
        program = hoist_mutation(program)

    # Enforce the depth limit
    if new_tree.tree_depth() >= d_limit:
        program = hoist_mutation(program)

    return program


def sf_mutation(parent, d_limit, s_limit):
    """ New 1 """
    if parent.tree.size_of_subtree() >= s_limit:
        return hoist_mutation(parent)

    # Get a random index in the range of the length of Parent A
    nodes = parent.tree.size_of_subtree()
    index = random.randint(0, nodes - 1)

    # Get a good subtree in Parent A to replace
    to_replace = subtree_at_index(parent.tree, index)

    # Get a random integer within the valid size range
    size_range = get_size_range(to_replace)
    new_size = random.randint(size_range[0], size_range[1])

    # Generate a new subtree using the grow method within the valid size range
    replacement = generate_tree_grow(new_size)

    # Replace the subtree in the parent's tree with the new tree we generated
    new_tree = replace_subtree_at_index(
        copy.deepcopy(parent.tree),
        index,
        replacement)

    # Create an individual with the new tree
    program = Individual(new_tree)

    # Try to conform to size and depth limits
    if new_tree.size_of_subtree() >= s_limit:
        program = hoist_mutation(program)

    if new_tree.tree_depth() >= d_limit:
        program = hoist_mutation(program)

    return program


# ==============================================================================
# ============================= Core GP Loop ===================================
# ==============================================================================


def core_gp_loop(max_iter, pop_size, rates_list, test_cases, t_size, s_limit=75, d_limit=14):
    """ The driving funciton that stops when reaching max generation count or
        finding a solution with zero error on the training cases. """

    # Initiate with random programs
    pop = [Individual(generate_random_program()) for i in range(pop_size)]

    # Track lowest error and best corresponding program
    best_t_error = MAX_FLOAT
    best_program = None

    # Start the loop of GP
    n_iter = 0
    while n_iter <= max_iter and best_t_error > 0:
        # Evaluate all candidates from the population, update best solution
        for program in pop:
            program.test(test_cases)
            if program.total_error < best_t_error:
                best_t_error = program.total_error
                best_program = program

        # Unpack the rates and prep children list
        x_rate, m_rate, r_rate = rates_list

        # Create a list for the next generation of programs
        children = []

        # Fill the population
        for _ in range(pop_size):
            # Get a random number
            choice = random.random()

            # Check if we need to do Crossover
            if choice < x_rate:
                # Parents Selection
                p1 = tournament_selection(random.sample(pop, t_size))
                p2 = tournament_selection(random.sample(pop, t_size))

                children.append(sf_crossover(p1, p2, d_limit, s_limit))

            # Check if we need to do Mutation
            elif choice < (x_rate + m_rate):
                p = tournament_selection(random.sample(pop, t_size))

                children.append(sf_mutation(p, d_limit, s_limit))

            # Hoist mutation
            else:
                p = tournament_selection(random.sample(pop, t_size))
                children.append(hoist_mutation(p))


        # Calculate the mean of some values
        mean = lambda values: sum(values) / len(values)

        # Get the depths, sizes, and errors of every individual
        depth, size, error = zip(*[
            (i.tree.tree_depth(),
            i.tree.size_of_subtree(),
            i.total_error) for i in pop])

        # Write the generation number to the outpout file
        file = open(OUTPUT, "a")
        file.write("GEN: " + str(n_iter) + '\n')

        # Write the average depth, size, and error for the generation
        file.write("Depth: {}\nSize: {}\nError: {}\n".format(
            mean(depth),
            mean(size),
            mean(error)))

        # Update the population with the new children we generated
        pop = children

        # Print the generation counter and increment the number of generations
        print(n_iter)
        n_iter += 1

        # Print the tree of the best program and the error of the best program
        print()
        print(best_program.tree)
        print(best_program.total_error)

        # Write a delimiter so we can easily read the output file
        file.write(("=" * 20) + '\n')
        file.close()

    return best_program, n_iter


# ==============================================================================
# ==================== Training Data and Main Functions ========================
# ==============================================================================

def read_data(filename, delimiter=",", has_header=True):
    """Reads classification data from a file.
    Returns a list of the header labels and a list containing each datapoint."""
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


def make_csv_training_cases(fname="192_vineyard.tsv"):
    """Makes a list of training cases. Each training case is a dictionary
    mapping x_i and y to their values. x_i starts at x0.

    Ex: {'x0': 1.0, 'x1': 5.0, 'y': 9.5}

    Right now, this is hard-coded to solve this problem:
    https://github.com/EpistasisLab/pmlb/tree/master/datasets/192_vineyard
    but it should be easy to adopt to other CSVs.
    """

    header, data = read_data(fname, "\t")
    cases = []

    for row in data:
        output = row[-1]
        inputs = row[:-1]
        row_dict = {"y": output}
        for i, input in enumerate(inputs):
            row_dict["x" + str(i)] = input
        cases.append(row_dict)

    return cases


def full_test():
    """ Run a full test of all the reasonable parameters """
    # Number of generations to run
    gens = [20, 50]

    # The maximum size of the population
    pop_sizes = [100, 500, 1000]

    # The rates of crossover, mutation, and hoist mutation
    rates = [[.7, .2, .1], [.2, .7, .1]]

    # All the test case files
    test_cases_list = ["project_test_func.tsv", "192_vineyard.tsv"]

    # All the tournament sizes
    t_sizes = [7]

    # The size limit on the programs
    s_limits = [75]

    # The depth limit on the programs
    d_limits = [7, 14]

    # Clear the contents of the output file
    file = open(OUTPUT, "w")
    file.write('')
    file.close()

    # For every maximum generation count
    for gen in gens:
        # For every population size
        for size in pop_sizes:
            # For every variation rate list
            for rate in rates:
                # For every tournament selection pool size
                for t_size in t_sizes:
                    # For every program soft size limit
                    for s_limit in s_limits:
                        # For every program soft depth limit
                        for d_limit in d_limits:
                            # For every dataset filename
                            for fname in test_cases_list:
                                # Write the parameters to the file
                                file = open(OUTPUT, "a")
                                file.write(
                                    "GENS: " + str(gen) + "\n" +
                                    "POPULATION SIZE: " + str(size) + "\n" +
                                    "RATES: " + str(rate) + "\n" +
                                    "TEST FILE: " + str(fname) + "\n" +
                                    "TOURNAMENT SIZE: " + str(t_size) + "\n" +
                                    "SIZE LIMIT: " + str(s_limit) + "\n" +
                                    "DEPTH LIMIT: " + str(d_limit) + "\n\n")

                                # Write a delimiter for the start of the run
                                file.write(('=' * 80) + '\n')

                                # Close the file to manage file handlers in the
                                # event that we crash during the run
                                file.close()

                                # Try to run with the set parameters
                                try:
                                    best_program, fin_iter = core_gp_loop(
                                        gen, # Generation count
                                        size, # Population size
                                        rate, # Mutation rates list
                                        make_csv_training_cases(fname),
                                        t_size, # Tournament selection pool size
                                        s_limit, # Soft program size limit
                                        d_limit) # Soft program depth limit

                                    # Append the results of the run to the file
                                    file = open(OUTPUT, "a")
                                    file.write(str(fin_iter) + '\n\n')
                                    file.write(str(best_program) + '\n')
                                    file.write(('=' * 40) + '\n')
                                    file.close()

                                    # Also print the results for monitoring
                                    # purposes in case the code randomly hangs
                                    print("GENS: " + str(gen) + "\n" +
                                    "POPULATION SIZE: " + str(size) + "\n" +
                                    "RATES: " + str(rate) + "\n" +
                                    "TEST FILE: " + str(fname) + "\n" +
                                    "TOURNAMENT SIZE: " + str(t_size) + "\n" +
                                    "SIZE LIMIT: " + str(s_limit) + "\n" +
                                    "DEPTH LIMIT: " + str(d_limit) + "\n\n")

                                # If there was an error print an error message
                                # and recover gracefully to continue the tests
                                except:
                                    print("RUN FAILED")


def main():
    # gens = 30                   # Maximum generation number
    # pop_size = 1000             # Population size
    # rates_list = [.5, .4, .1]   # Size-fair crossover/mutation/hoist mutation
    # test_cases = make_csv_training_cases("project_test_func.tsv")
    # t_size = 7                      # Tournament size
    #
    # best_program, fin_iter = core_gp_loop(gens, pop_size, rates_list, test_cases, t_size)
    # # print("best_program:", best_program)
    # # print("total iteration:", fin_iter)
    # print("end error", best_program)

    full_test()


if __name__ == "__main__":
    main()
