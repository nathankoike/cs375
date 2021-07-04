"""
HW 3
GP Implementation start

Implements nodes for program trees and random initialization.
"""

import operator, random, math

MAX_FLOAT = 1e12

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

# Dictionary mapping function stings to functions that perform them
FUNCTION_DICT = {"+": operator.add,
                 "-": operator.sub,
                 "*": operator.mul,
                 "/": safe_division,
                 "exp": safe_exp,
                 "sin": math.sin,
                 "cos": math.cos}

# Dictionary mapping function strings to their arities (number of arguments)
FUNCTION_ARITIES = {"+": 2,
                    "-": 2,
                    "*": 2,
                    "/": 2,
                    "exp": 1,
                    "sin": 1,
                    "cos": 1}

# List of function strings
FUNCTIONS = list(FUNCTION_DICT.keys())

# Strings for each variable
VARIABLES = ["x1", "x2"]




def main():

    print(safe_division(45, 0))


if __name__ == "__main__":
    main()
