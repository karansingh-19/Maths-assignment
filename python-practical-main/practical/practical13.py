# Practical 13 :

import sympy as sp
import numpy as np

# ---------- Symbols ----------
x, y, z = sp.symbols('x y z')

# ---------- Curl of a vector field ----------
def curl(F, variables):
    # curl F = (dF3/dy - dF2/dz) i + (dF1/dz - dF3/dx) j + (dF2/dx - dF1/dy) k
    X, Y, Z = variables
    F1, F2, F3 = F
    return sp.Matrix([sp.diff(F3, Y) - sp.diff(F2, Z),
                      sp.diff(F1, Z) - sp.diff(F3, X),
                      sp.diff(F2, X) - sp.diff(F1, Y)])

def divergence(F, variables):
    return sum(sp.diff(Fi, v) for Fi, v in zip(F, variables))

vars3 = (x, y, z)


# ---------- 1. Curl of a vector field ----------
print("=== 1. Curl of F = (x^2*y, y^2*z, z^2*x) ===")
F1 = [x**2 * y, y**2 * z, z**2 * x]
c1 = curl(F1, vars3)
print("F =", F1)
print("curl F =", c1.T)
print("curl F at (1, 2, 3) =", c1.subs({x: 1, y: 2, z: 3}).T)


# ---------- 2. Rotational field ----------
print("\n=== 2. Curl of F = (-y, x, 0) ===")
F2 = [-y, x, 0]
print("curl F =", curl(F2, vars3).T)
print("(non-zero, so the field is rotational)")


# ---------- 3. Curl of the position vector r ----------
print("\n=== 3. Curl of position vector r = (x, y, z) ===")
print("curl r =", curl([x, y, z], vars3).T)


# ---------- 4. Irrotational check and scalar potential ----------
print("\n=== 4. Is F = (y*z, z*x, x*y) irrotational? ===")
F4 = [y * z, z * x, x * y]
c4 = curl(F4, vars3)
print("curl F =", c4.T)
print("Irrotational (curl F = 0)?:", c4.is_zero_matrix)

phi4 = x * y * z                             # scalar potential
grad_phi4 = [sp.diff(phi4, v) for v in vars3]
print("Scalar potential phi = x*y*z, grad(phi) =", grad_phi4)
print("grad(phi) = F ?:", all(sp.simplify(a - b) == 0 for a, b in zip(grad_phi4, F4)))


# ---------- 5. Curl of a gradient is zero ----------
print("\n=== 5. Verify curl(grad phi) = 0 for phi = x^2*y + y*z^3 ===")
phi = x**2 * y + y * z**3
grad_phi = [sp.diff(phi, v) for v in vars3]
print("grad(phi) =", grad_phi)
c5 = curl(grad_phi, vars3)
print("curl(grad phi) =", c5.T)
print("Identity verified?:", c5.is_zero_matrix)


# ---------- 6. Divergence of a curl is zero ----------
print("\n=== 6. Verify div(curl F) = 0 for F = (x^2*y, y^2*z, z^2*x) ===")
print("curl F =", c1.T)
print("div(curl F) =", sp.simplify(divergence(list(c1), vars3)))


# ---------- 7. Numerical check using NumPy (central difference) ----------
print("\n=== 7. Numerical check using NumPy ===")
def F_num(p):
    X, Y, Z = p
    return np.array([X**2 * Y, Y**2 * Z, Z**2 * X])

def num_curl(F, p, h=1e-5):
    p = np.array(p, dtype=float)
    J = np.zeros((3, 3))                     # J[i, j] = dFi/dxj
    for j in range(3):
        step = np.zeros(3)
        step[j] = h
        J[:, j] = (F(p + step) - F(p - step)) / (2 * h)
    return np.array([J[2, 1] - J[1, 2],
                     J[0, 2] - J[2, 0],
                     J[1, 0] - J[0, 1]])

nc = num_curl(F_num, [1, 2, 3])
sym = np.array(c1.subs({x: 1, y: 2, z: 3}).T.tolist()[0], dtype=float)
print("Numerical curl F at (1, 2, 3) =", np.round(nc, 5))
print("Matches symbolic result?:", np.allclose(nc, sym, atol=1e-4))