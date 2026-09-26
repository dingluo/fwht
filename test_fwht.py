"""Check both transforms against the sequency-ordered Hadamard matrix."""
import numpy as np

import FWHT


def hadamard(n):
    block = np.array([[1.0]])
    while block.shape[0] < n:
        block = np.block([[block, block], [block, -block]])
    return block


def sequency_order(matrix):
    sign_changes = [np.sum(np.abs(np.diff(row))) for row in matrix]
    return np.argsort(sign_changes)


def check(n):
    matrix = hadamard(n)[sequency_order(hadamard(n))]
    values = np.arange(n, dtype=float)
    expected = matrix @ values / n
    fast = FWHT.FWHT(values).ravel()
    slow = FWHT.SFWHT(values)
    assert np.allclose(fast, expected), (n, "FWHT")
    assert np.allclose(slow, expected), (n, "SFWHT")
    assert np.allclose(fast, slow), (n, "FWHT vs SFWHT")


if __name__ == "__main__":
    for length in (8, 16, 32, 64):
        check(length)
    print("ok")
