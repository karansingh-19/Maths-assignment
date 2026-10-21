# PRACTICAL 1 :
    
import numpy as np

# ---------- Creating vectors ----------
v = np.array([1+2j, 3-1j, 4+0j])          # row vector (complex)
col_v = v.reshape(-1, 1)                  # column vector

print("Vector v:", v)
print("Column vector:\n", col_v)

# ---------- Creating matrices ----------
A = np.array([[1+1j, 2-1j, 3],
              [4,    5+2j, 6-3j]])

print("\nMatrix A:\n", A)
print("Shape of A:", A.shape)

# ---------- Transpose ----------
# Vector: a 1-D array's transpose is unchanged, so convert to column first
print("\nTranspose of row vector (as column):\n", v.reshape(1, -1).T)
print("Transpose of column vector (as row):\n", col_v.T)

# Matrix
print("\nTranspose of A:\n", A.T)

# ---------- Conjugate ----------
print("\nConjugate of v:", np.conj(v))
print("Conjugate of A:\n", np.conj(A))

# ---------- Conjugate transpose (Hermitian) ----------
# Vector
print("\nConjugate transpose of v:\n", np.conj(v.reshape(1, -1)).T)

# Matrix
A_H = A.conj().T          # same as np.conjugate(A).T
print("Conjugate transpose of A:\n", A_H)
print("Shape of A^H:", A_H.shape)

# ---------- Verification: (A^H)^H = A ----------
print("\n(A^H)^H equals A:", np.array_equal(A_H.conj().T, A))

# ---------- Real matrix: transpose = conjugate transpose ----------
B = np.array([[1, 2, 3],
              [4, 5, 6]])
print("\nReal matrix B:\n", B)
print("B^T:\n", B.T)
print("B^H:\n", B.conj().T)
print("B^T == B^H:", np.array_equal(B.T, B.conj().T))

# ---------- Hermitian check on a square matrix ----------
H = np.array([[2, 1-1j],
              [1+1j, 3]])
print("\nIs H Hermitian (H == H^H)?", np.allclose(H, H.conj().T))