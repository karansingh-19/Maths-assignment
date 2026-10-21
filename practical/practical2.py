# PRACTICAL 2 :

#Echelon Form Of Matrix :
import sympy as sp

# ---------- Method 1: Using SymPy (gives the actual echelon form) ----------
A = sp.Matrix([[1, 2, 3, 4],
               [2, 4, 7, 8],
               [3, 6, 10, 12]])

print("Original matrix A:")
sp.pprint(A)

# Row echelon form (rref gives the REDUCED row echelon form)
rref_matrix, pivot_cols = A.rref()
print("\nReduced row echelon form:")
sp.pprint(rref_matrix)
print("Pivot columns:", pivot_cols)

# Rank = number of pivots (non-zero rows in echelon form)
print("\nRank of A (SymPy):", A.rank())
print("Rank from pivots:", len(pivot_cols))