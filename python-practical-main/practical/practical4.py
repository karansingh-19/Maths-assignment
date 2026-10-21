# Practical 4 :

import numpy as np

# ---------- Gauss elimination (forward elimination on augmented matrix) ----------
def gauss_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    M = np.hstack((A, b))                    # augmented matrix [A | b]
    n, m = A.shape
    r = 0                                    # current pivot row
    pivots = []
    for c in range(m):
        if r >= n:
            break
        p = np.argmax(np.abs(M[r:, c])) + r  # partial pivoting
        if np.isclose(M[p, c], 0):
            continue                         # no pivot in this column
        M[[r, p]] = M[[p, r]]                # swap rows
        for i in range(r + 1, n):
            M[i] = M[i] - (M[i, c] / M[r, c]) * M[r]
        pivots.append(c)
        r += 1
    M[np.isclose(M, 0)] = 0                  # clean tiny values
    rank_A = len(pivots)
    rank_Ab = np.sum(np.any(~np.isclose(M, 0), axis=1))
    return M, rank_A, rank_Ab


def solve_system(A, b):
    M, rank_A, rank_Ab = gauss_elimination(A, b)
    n = len(A[0])
    print("Echelon form of [A|b]:\n", M)
    print("rank(A) =", rank_A, ", rank([A|b]) =", rank_Ab)

    if rank_A < rank_Ab:
        print("Result: No solution (inconsistent system)")
    elif rank_A == n:
        # back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (M[i, -1] - M[i, i + 1:n] @ x[i + 1:n]) / M[i, i]
        print("Result: Unique solution x =", np.round(x, 6))
    else:
        print("Result: Infinitely many solutions, free variables =", n - rank_A)


# ---------- 1. Non-homogeneous system AX = B ----------
print("=== Non-homogeneous system ===")
A1 = [[2, 1, -1],
      [-3, -1, 2],
      [-2, 1, 2]]
b1 = [8, -11, -3]
solve_system(A1, b1)

# ---------- 2. Homogeneous system AX = 0 (non-trivial solution) ----------
print("\n=== Homogeneous system (non-trivial solution) ===")
A2 = [[1, 1, 1],
      [2, 3, 1],
      [3, 4, 2]]
b2 = [0, 0, 0]
solve_system(A2, b2)