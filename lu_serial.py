# A = L U de N x N, con L triangular inferior y U triangular superior.
# El resultado se guarda en el mismo arreglo: U en el triangulo superior
# (incluida la diagonal) y L bajo la diagonal (su diagonal de unos no se guarda).
# Uso: python lu_serial.py [--N N] [--b b] [--check]

import argparse
import time

import numpy as np
from scipy.linalg import lu_factor, solve_triangular


def matrix_entries(rows, cols, N):
    """Entradas A[rows, cols] de la matriz del problema, a partir de los indices globales.

    Fuera de la diagonal los valores estan en el intervalo [-1, 1] y la diagonal vale N, de modo que A
    es estrictamente diagonal dominante por filas y por columnas. Cada proceso puede
    generar solo sus bloques, y la matriz no depende de la distribucion ni de b.
    """
    I = np.asarray(rows, dtype=np.float64)[:, None]
    J = np.asarray(cols, dtype=np.float64)[None, :]
    A = np.sin(1e-3 * I * J + 0.3 * I + 0.7 * J)
    A[np.asarray(rows)[:, None] == np.asarray(cols)[None, :]] = N
    return A


def lu_diag_block(Akk):
    """LU del bloque diagonal, en el mismo arreglo.

    Como A es diagonal dominante, piv no intercambia filas.
    """
    lu, piv = lu_factor(Akk, overwrite_a=True, check_finite=False)
    assert np.all(piv == np.arange(len(piv))), "Hay pivoteo."
    Akk[:] = lu


def lu_blocked(A, b):
    """Factorizacion LU por bloques (right-looking), en el mismo arreglo."""
    N = A.shape[0]
    for k in range(0, N, b):
        e = min(k + b, N)
        # 1. Factorizar el bloque diagonal: A_kk = L_kk U_kk.
        lu_diag_block(A[k:e, k:e])
        Lkk = A[k:e, k:e]
        # 2. Panel de U: U_kj = L_kk^{-1} A_kj.
        A[k:e, e:] = solve_triangular(Lkk, A[k:e, e:], lower=True,
                                      unit_diagonal=True, check_finite=False)
        # 3. Panel de L: L_ik = A_ik U_kk^{-1}.
        A[e:, k:e] = solve_triangular(Lkk, A[e:, k:e].T, trans="T", lower=False,
                                      check_finite=False).T
        # 4. Actualizar la submatriz restante: A_ij -= L_ik U_kj.
        A[e:, e:] -= A[e:, k:e] @ A[k:e, e:]


def residual(A, LU):
    """Error relativo ||A - L U||_F / ||A||_F."""
    L = np.tril(LU, -1) + np.eye(LU.shape[0]) 
    U = np.triu(LU)
    return np.linalg.norm(A - L @ U) / np.linalg.norm(A)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--b", type=int, default=256)
    parser.add_argument("--check", action="store_true",
                        help="calcula ||A - LU|| / ||A||")
    args = parser.parse_args()
    N, b = args.N, args.b

    idx = np.arange(N)
    A = matrix_entries(idx, idx, N)
    A0 = A.copy() if args.check else None

    t0 = time.perf_counter()
    lu_blocked(A, b)
    t1 = time.perf_counter()

    gflops = 2 * N**3 / 3 / (t1 - t0) / 1e9
    print(f"N={N} b={b} tiempo={t1 - t0:.3f} s GFLOP/s={gflops:.1f} "
          f"checksum={A.sum():.6e}")
    if args.check:
        print(f"residuo relativo={residual(A0, A):.3e}")


if __name__ == "__main__":
    main()
