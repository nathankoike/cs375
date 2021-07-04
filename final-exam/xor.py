import math

tests = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]

def sigmoid(x):
    """Logistic / signmoid function."""

    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0

    return 1.0 / denom

def fprop(vals):
    [a, b, c, d, e, f] = vals

    # For every test
    for (x, y) in tests:
        # Get the prediction
        calc = round(
            sigmoid(
                (e * sigmoid((a * x[0]) + (b * x[1]))) +
                (f * sigmoid((c * x[0]) + (d * x[1]))))
        )

        # If there is a mismatched output
        if calc != y:
            return False

    return True

min_sol = [9999999999, None]

# For all the possible values of [a, b, c, d, e, f]
for a in range(-10, 11):
    for b in range(-10, 11):
        for c in range(-10, 11):
            for d in range(-10, 11):
                for e in range(-10, 11):
                    for f in range(-10, 11):
                        valid = fprop([a, b, c, d, e, f])

                        # If this is a valid solution
                        if valid:
                            # If this is a minimal absolute value sum solution
                            if sum([abs(a), abs(b), abs(c), abs(d), abs(e), abs(f)]) < min_sol[0]:
                                min_sol = [
                                    sum([abs(a), abs(b), abs(c), abs(d), abs(e), abs(f)]),
                                    [a, b, c, d, e, f]]

# Print the sum of the absolute values of the weights as well as the weights
print('\n\n\n')
print(min_sol)
print()

# Get the raw output
[a, b, c, d, e, f] = min_sol[1]
for (x, y) in tests:
    print(sigmoid(
        (e * sigmoid((a * x[0]) + (b * x[1]))) +
        (f * sigmoid((c * x[0]) + (d * x[1])))))
