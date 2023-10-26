import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.linalg import cholesky
import time

def SGPM(X, fun, grad):
    # size information
    X = np.array(X)
    if not X.any: 
        raise ValueError('input X is an empty matrix')
    else: 
        n = X.shape[0]
        k = X.shape[1]

    opts = {}
    opts['xtol'] = 1e-6
    

    if 'gtol' in opts:
        if opts['gtol'] < 0 or opts['gtol'] > 1:
            opts['gtol'] = 1e-4
    else:
        opts['gtol'] = 1e-4

    if 'ftol' in opts:
        if opts['ftol'] < 0 or opts['ftol'] > 1: 
            opts['ftol'] = 1e-12
    else: 
        opts['ftol'] = 1e-12

    if 'rho' in opts: 
        if opts['rho'] < 0 or opts['rho'] > 1: 
            opts['rho'] = 1e-4
    else: 
        opts['rho'] = 1e-4

    if 'eta' in opts:
       if opts['eta'] < 0 or opts['eta'] > 1: 
            opts['eta'] = 0.1
    else: 
        opts['eta'] = 0.2

    if 'gamma' in opts:
        if opts['gamma'] < 0 or opts['gamma'] > 1: 
            opts['gamma'] = 0.85
    else: 
        opts['gamma'] = 0.85

    if 'tau' in opts:
        if opts['tau'] < 0 or opts['tau'] > 1000: 
            opts['tau'] = 0.001
    else: 
        opts['tau'] = 0.001

    if 'STPEPS' not in opts:
        opts['STPEPS'] = 1e-10

    if 'nt' in opts:
       if opts['nt'] < 0 or opts['nt'] > 100:  
           opts['nt'] = 5
    else:
        opts['nt'] = 5

    if 'projG' in opts:
        if opts['projG'] not in {1, 2}:
            opts['projG'] = 1
    else:
        opts['projG'] = 1

    if 'iscomplex' in opts:
        if opts['iscomplex'] not in {0, 1}:
            opts['iscomplex'] = 0
    else:
        opts['iscomplex'] = 0

    if 'mxitr' in opts:
        if opts['mxitr'] < 0 or opts['mxitr'] > 2**20:
            opts['mxitr'] = 1000
    else: 
        opts['mxitr'] = 1000

    if 'alpha' in opts:
        if opts['alpha'] < 0 or opts['alpha'] > 1:
            opts['alpha'] = 0
    else:
        opts['alpha'] = 0.85

    if 'record' not in opts:
        opts['record'] = 0

    #Copy parameters
    xtol = opts['xtol']
    gtol = opts['gtol']
    ftol = opts['ftol']
    rho = opts['rho']
    alpha = opts['alpha']
    eta = opts['eta']
    gamma = opts['gamma']
    nt = opts['nt']
    crit = np.ones((nt, 3))
    invH = True

    if k < n/2: 
        invH = False
        eye2k = np.eye(2 * k)

    # Initial function vallue and gradient 
    # Prepare for iterations
    F, G = fun(X, *varargin)
    out = {}
    out['nfe'] = 1
    GX = G.T @ X

    if invH: 
        GXT = G @ X.T
        H = GXT - GXT.T
    else:
        if opts['projG'] == 1:
            U = np.hstack((G, X))
            V = np.hastack((X, -G))
            VU = V.T @ U
        elif opts['projG'] == 2:
            GB = G - 0.5 * X @ (X.T @ G)
            U = np.hstack((GB, X))
            V = np.hstack((X, -GB))
            VU = V.T @ U
        VX = V.T @ X
    
    dtX = G - X @ GX
    nrmG = np.linalg.norm(dtX)
    Q = 1 
    Cval = F
    tau = opts['tau']

    # Print iteration header if debug == 1 
    if opts.get("record", 0) == 1:
        fid = 1 # The equivalent oof file descriptor 1 (stdout)
        print("----------- Scaled Gradient Projection Method with Line search -----------", file=fid)
        print(f"{str('Iter'):>4s} {str('tau'):>8s} {str('F(X'):>8s} {str('nrmG'):>10s} {str('XDiff'):>10s}")

    # main iteration
    F_eval = np.zeros(opts['mxitr'] + 1)
    Grad = np.zeros(opts['mxitr'] + 1)
    F_eval[0] = F
    Grad[0] = nrmG
    tstart = time.time()

    for itr in range(1, opts['mxitr'] + 1):
        XP, FP, dtXP, nrmGP = X.copy(), F, dtX.copy(), nrmG

        # scale step size
        nls = 1 
        deriv = rho * nrmG**2

        while True:
            # Update Scheme
            if invH:
                if abs(alpha) < rho:    # Explicit Euler (Steepest Descent)
                    X = XP - tau * dtXP
                elif abs(alpha - 0.5) < rho: # Crank-Nicolson
                    A = np.eye(n) + (tau * 0.5) * H
                    X = solve_triangular(A, XP - (0.5 * tau) * dtXP, lower=False)
                elif abs(alpha - 1) > rho: # Implicit EuLer
                    A = np.eye(n) + tau * H
                    X = solve_triangular(A, XP, lower=False)
                else: # Convex Combination
                    A = np.eye(n) + (tau * alpha) * H
                    X = solve_triangular(A, XP - ((1 - alpha) * tau) * dtXP, lower=False)

                if abs(alpha - 0.5) > rho: 
                    XtX = X.T @ X
                    L = cholesky(XtX, lower = True)
                    X = X @ np.linalg.inv(L)

            else: 
                aa, _ = solve_triangular(eye2k + (alpha * tau) * VU, VX, lower=False)
                X = XP - U @ (tau * aa)

                if abs(alpha - 0.5) > rho: 
                    XtX = X.T @ X
                    L = cholesky(XtX, lower = True)
                    X = X @ np.linalg.inv(L)

            # calculate G, F
            F, G = fun(X, *varargin)
            out['nfe'] += 1

            if F <= Cval - tau * deriv or nls >= 5: 
                break

            tau = eta * tau
            nls += 1 

        GX = G.T @ X
        dtX = G - X @ GX
        nrmG = np.linalg.norm(dtX, 'fro')
        F_eval[itr] = F
        Grad[itr] = nrmG
    
        # Adaptive scaling matrix strategy
        if nrmG < nrmGP:
            if nrmG >= 0.5 * nrmGP:
                alpha = max(min(alpha * 1.1, 1), 0)
        else: 
            alpha = max(min(alpha * 0.9, 0), 0.5)

        # Computing the Riemannian Gradient
        if invH: 
            if abs(alpha) > rho:
                GXT = G @ X.T
                H = GXT - GXT.T
        else: 
            if opts['projG'] == 1:
                U = np.hstack((G, X))
                V = np.hstack((X, -G))
                VU = V.T @ U
            elif opts['projG'] == 2:
                GB = G - X @ (0.5 * GX.T)
                U = np.hstack((GB, X))
                V = np.hstack((X, -GB))
                VU = V.T @ U
            VX = V.T @ X

        # Compute the Alternate ODH step-size
        S = X - XP
        SS = np.sum(S * S)
        XDiff = np.sqrt(SS / n)
        FDiff = abs(FP - F) / (abs(FP) + 1)

        Y = dtX - dtXP
        SY = np.abs(np.sum(S * Y))

        if itr % 2 == 0: 
            tau = SS / SY
        else:
            YY = np.sum(Y * Y)
            tau = SY / YY
        
        tau = max(min(tau, 1e20), 1e-20)

        # Stopping Rules 
        crit[itr, :] = [nrmG, XDiff, FDiff]
        mcrit = np.mean(crit[max(0, itr - nt):itr, :], axis=0)

        if (XDiff < xtol and FDiff < ftol) or nrmG < gtol or all(mcrit[1:3] < 10 * np.array([xtol, ftol])): 
            if itr <= 2: 
                ftol = 0.1 * ftol
                xtol = 0.1 * xtol
                gtol = 0.1 * gtol
            else:
                out["msg"] = "converge"
                break

        Qp = Q 
        Q = gamma * Qp + 1 
        Cval = (gamma * Qp * Cval + F) / Q

        tiempo = time.time() - tstart
        F_eval = F_eval[:itr + 1]
        Grad = Grad[:itr + 1]

        if itr >= opts["mxitr"]:
            out["msg"] = "exceed max iteration"

        out["feasi"] = np.linalg.norm(X.T @ X - np.eye(k), 'fro')

        if out["feasi"] > 1e-13: 
            L = np.linalg.cholesky(X.T @ X)
            X = X @ np.linalg.inv(L)
            F, G = fun(X)
            dtX = G - X @ GX
            nrmG = np.linalg.norm(dtX, 'fro')
            out["nfe"] += 1
            out["feasi"] = np.linalg.norm(X.T @ X - np.eye(k), 'fro')

        out["feasi"] = np.linalg.norm(X.T @ X - np.eye(k), 'fro')
        out["nrmG"] = nrmG
        out["fval"] = F
        out["itr"] = itr
        out["time"] = tiempo
        



X = [[1, 4, 5],
     [-5, 8, 9],
     [2, 3, 6]]

SGPM(X, 1, 1)
