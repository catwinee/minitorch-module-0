"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b


def id(a: float) -> float:
    """Returns the input unchanged."""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negates a number."""
    return -a


def lt(a: float, b: float) -> bool:
    """Checks if one number is less than another."""
    return a < b


def eq(a: float, b: float) -> bool:
    """Checks if two numbers are equal."""
    return a == b


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers."""
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Checks if two numbers are close in value.

    Returns:
        True if the absolute value of the two numbers is less than 0.01.

    """
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function.

    Returns:
        The output of the sigmoid function, a float value between 0 and 1.

    """
    return 1.0 / (1.0 + math.exp(-abs(a)))


def relu(a: float) -> float:
    """Applies the ReLU activation function.

    Returns:
        The input if it is positive, otherwise 0.

    """
    return max(0, a)


def log(a: float) -> float:
    """Calculates the natural logarithm.

    Args:
        a: A non-zero number.

    Raises:
        ValueError: if the input is not positive.

    """
    if a <= 0:
        raise ValueError("Can not compute the logarithm of a non-positive number")
    return math.log(a)


def exp(a: float) -> float:
    """Calculates the exponential function."""
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal.

    Args:
        a (float): A non-zero number.

    Raises:
        ValueError: if the input is zero.

    """
    if a == 0:
        raise ValueError("Can not compute the reciprocal of zero")
    return 1.0 / a


def log_back(a: float, d: float) -> float:
    """Computes the derivative of log times a second arg.

    Args:
        a (float): The input to the original log function.
        d (float): The upstream gradient.

    Returns:
        The downstream gradient d / a.

    """
    return d / a


def inv_back(a: float, d: float) -> float:
    """Computes the derivative of reciprocal times a second arg.

    Args:
        a (float): The input to the original inv function.
        d (float): The upstream gradient.

    Returns:
        The downstream gradient -d * a ** -2.

    """
    return -d * a**-2


def relu_back(a: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
        a (float): The input to the original relu function.
        d (float): The upstream gradient.

    Returns:
        The downstream gradient. 0 if the input is negative, d otherwise.

    """
    return 0 if a < 0 else d


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map() -> Callable:
    """TODO."""
    raise NotImplementedError("Not implemented.")


def zipWith() -> Callable:
    """TODO."""
    raise NotImplementedError("Not implemented.")


def reduce() -> Callable:
    """TODO."""
    raise NotImplementedError("Not implemented.")


def negList() -> Iterable[float]:
    """Negate a list."""
    raise NotImplementedError("Not implemented.")


def addLists() -> Iterable[float]:
    """Add two lists together."""
    raise NotImplementedError("Not implemented.")


def sum() -> Iterable[float]:
    """Sum lists."""
    raise NotImplementedError("Not implemented.")


def prod() -> Iterable[float]:
    """Take the product of lists."""
    raise NotImplementedError("Not implemented.")
