# Practical 3 :

import sympy as sp

# ---------- Method 1: Using SymPy (exact results) ----------
A = sp.Matrix([[1, 2, 3],
               [0, 1, 4],
               [5, 6, 0]])

print("Matrix A:", A)
print("Determinant:", A.det())
print("Cofactor matrix:", A.cofactor_matrix())
print("Adjoint (adjugate):", A.adjugate())

if A.det() != 0:
    print("Inverse:", A.inv())
else:
    print("Inverse does not exist (matrix is singular)")