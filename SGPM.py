# import numpy as np
# import pandas as pd
# from scipy.linalg import solve_triangular
# from scipy.linalg import cholesky
# import time


# def SGPM(X, A, B, At, Bt, a):
#     # size information
#     X = np.array(X)
#     n = X.shape[0]
#     k = X.shape[1]

#     xtol = 1e-6
#     gtol = 1e-4
#     ftol = 1e-12
#     rho = 1e-4
#     eta = 0.1
#     gamma = 0.85
#     tau = 1e-3
#     STPEPS = 1e-10
#     nt = 5
#     mxitr = 1000
#     alpha = 0.85
#     record = 0
#     projG = 1
#     crit = np.ones((nt, 3))
#     invH = True

#     if k < n/2:
#         invH = False
#         eye2k = np.eye(2 * k)

#     # Initial function value and gradient
#     # Prepare for iterations
#     F = f(A, B, a, X, At, Bt)
#     G = grad(A, B, a, X, At, Bt)
#     out = {}
#     out['nfe'] = 1
#     GX = G.T @ X

#     if invH:
#         GXT = G @ X.T
#         H = GXT - GXT.T
#     # else:
#     #     if projG == 1:
#     #         U = np.hstack((G, X))
#     #         V = np.hastack((X, -G))
#     #         VU = V.T @ U
#     #     elif projG == 2:
#     #         GB = G - 0.5 * X @ (X.T @ G)
#     #         U = np.hstack((GB, X))
#     #         V = np.hstack((X, -GB))
#     #         VU = V.T @ U
#     #     VX = V.T @ X

#     dtX = G - X @ GX
#     nrmG = np.linalg.norm(dtX)
#     Q = 1
#     Cval = F

#     # main iteration
#     # F_eval = np.zeros(mxitr + 1)
#     # Grad = np.zeros(mxitr + 1)
#     F_eval = F
#     Grad = nrmG
#     # F_eval[0] = F
#     # Grad[0] = nrmG
#     tstart = time.time()

#     for itr in range(1, mxitr + 1):
#         XP, FP, dtXP, nrmGP = X.copy(), F, dtX.copy(), nrmG

#         # scale step size
#         nls = 1
#         deriv = rho * nrmG**2

#         while True:
#             # Update Scheme
#             if invH:
#                 if abs(alpha) < rho:    # Explicit Euler (Steepest Descent)
#                     X = XP - tau * dtXP
#                 elif abs(alpha - 0.5) < rho:  # Crank-Nicolson
#                     A = np.eye(n) + (tau * 0.5) * H
#                     X = solve_triangular(
#                         A, XP - (0.5 * tau) * dtXP, lower=False)
#                 elif abs(alpha - 1) > rho:  # Implicit EuLer
#                     A = np.eye(n) + tau * H
#                     X = solve_triangular(A, XP, lower=False)
#                 else:  # Convex Combination
#                     A = np.eye(n) + (tau * alpha) * H
#                     X = solve_triangular(
#                         A, XP - ((1 - alpha) * tau) * dtXP, lower=False)

#                 if abs(alpha - 0.5) > rho:
#                     XtX = X.T @ X
#                     L = cholesky(XtX, lower=True)
#                     X = X @ np.linalg.inv(L)

#             # else:
#             #     aa, _ = solve_triangular(
#             #         eye2k + (alpha * tau) * VU, VX, lower=False)
#             #     X = XP - U @ (tau * aa)

#             #     if abs(alpha - 0.5) > rho:
#             #         XtX = X.T @ X
#             #         L = cholesky(XtX, lower=True)
#             #         X = X @ np.linalg.inv(L)

#             # calculate G, F
#             F = f(A, B, a, X, At, Bt)
#             G = grad(A, B, a, X, At, Bt)
#             out['nfe'] += 1

#             if F <= Cval - tau * deriv or nls >= 5:
#                 break

#             tau = eta * tau
#             nls += 1

#         GX = G.T @ X
#         dtX = G - X @ GX
#         nrmG = np.linalg.norm(dtX, 'fro')
#         F_eval = F
#         Grad = nrmG

#         # Adaptive scaling matrix strategy
#         if nrmG < nrmGP:
#             if nrmG >= 0.5 * nrmGP:
#                 alpha = max(min(alpha * 1.1, 1), 0)
#         else:
#             alpha = max(min(alpha * 0.9, 0), 0.5)

#         # Computing the Riemannian Gradient
#         if invH:
#             if abs(alpha) > rho:
#                 GXT = G @ X.T
#                 H = GXT - GXT.T
#         else:
#             if projG == 1:
#                 U = np.hstack((G, X))
#                 V = np.hstack((X, -G))
#                 VU = V.T @ U
#             elif projG == 2:
#                 GB = G - X @ (0.5 * GX.T)
#                 U = np.hstack((GB, X))
#                 V = np.hstack((X, -GB))
#                 VU = V.T @ U
#             VX = V.T @ X

#         # Compute the Alternate ODH step-size
#         S = X - XP
#         SS = np.sum(S * S)
#         XDiff = np.sqrt(SS / n)
#         FDiff = abs(FP - F) / (abs(FP) + 1)

#         Y = dtX - dtXP
#         SY = np.abs(np.sum(S * Y))

#         if itr % 2 == 0:
#             tau = SS / SY
#         else:
#             YY = np.sum(Y * Y)
#             tau = SY / YY

#         tau = max(min(tau, 1e20), 1e-20)

#         # Stopping Rules
#         crit[itr, :] = [nrmG, XDiff, FDiff]
#         mcrit = np.mean(crit[max(0, itr - nt):itr, :], axis=0)

#         if (XDiff < xtol and FDiff < ftol) or nrmG < gtol or all(mcrit[1:3] < 10 * np.array([xtol, ftol])):
#             if itr <= 2:
#                 ftol = 0.1 * ftol
#                 xtol = 0.1 * xtol
#                 gtol = 0.1 * gtol
#             else:
#                 out["msg"] = "converge"
#                 break

#         Qp = Q
#         Q = gamma * Qp + 1
#         Cval = (gamma * Qp * Cval + F) / Q

#         tiempo = time.time() - tstart
#         F_eval = F_eval
#         Grad = Grad

#         if itr >= mxitr:
#             out["msg"] = "exceed max iteration"

#         out["feasi"] = np.linalg.norm(X.T @ X - np.eye(k), 'fro')

#         if out["feasi"] > 1e-13:
#             L = np.linalg.cholesky(X.T @ X)
#             X = X @ np.linalg.inv(L)
#             F = f(A, B, a, X, At, Bt)
#             G = grad(A, B, a, X, At, Bt)
#             dtX = G - X @ GX
#             nrmG = np.linalg.norm(dtX, 'fro')
#             out["nfe"] += 1
#             out["feasi"] = np.linalg.norm(X.T @ X - np.eye(k), 'fro')

#         out["feasi"] = np.linalg.norm(X.T @ X - np.eye(k), 'fro')
#         out["nrmG"] = nrmG
#         out["fval"] = F
#         out["itr"] = itr
#         out["time"] = tiempo

#         print(X)
