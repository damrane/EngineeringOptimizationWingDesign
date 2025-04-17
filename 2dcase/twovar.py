"""
This is a simple example of a constrained optimization problem using
augmented Lagrangian method with finite-difference gradients and Hessians.
"""

import numpy as np

def objective(x):
    ct, cr= x
    beff = 50
    return (ct + cr)/(2*beff)

def constraints(x):
    """Return array g(x) for the 12 inequalities g_i(x) <= 0."""
    ct, cr = x
    beff = 50
    S = ct + cr
    taper = ct/cr
    halfchord = 30*np.pi/180
    sqrt_t = np.sqrt(1 - np.sin(halfchord)**2)

    return np.array([
        taper - 0.31,                 
        -taper + 0.21,                
        2*beff/S - 9.5,               
        -2*beff/S + 7.5,              
        (2/3)*cr*((1+taper+taper**2)/(1+taper)) - 12,  
        -((2/3)*cr*((1+taper+taper**2)/(1+taper))) + 4,
        beff*S/(2*sqrt_t) - 820,      
        -beff*S/(2*sqrt_t) + 120,                   
        -ct,                          
        -cr                           
    ])

def aug_lagrangian(x, mu, rho):
    """Augmented Lagrangian function: L(x, λ) = f(x) + Σ [ λ_i * g_i(x) + 0.5 * ρ * g_i(x)^2 ]"""
    f0 = objective(x)
    g = constraints(x)
    # only penalize positive violations:
    penalty = np.sum(np.maximum(0, g)**2)
    return f0 + np.dot(mu, g) + 0.5*rho*penalty

def grad_L(Lfun, x, h=1e-8):
    """Approximation of Lagrangian gradient using finite-difference method."""
    n = len(x)
    f0 = Lfun(x)
    grad = np.zeros(n)
    for i in range(n):
        xh = x.copy()
        xh[i] += h
        grad[i] = (Lfun(xh) - f0)/h
    return grad

def hess_L(fun, x, h=1e-5):
    """Approximation of Lagrangian Hessian using finite-difference method."""
    n = len(x)
    H = np.zeros((n,n))
    g0 = grad_L(fun, x, h=h/10)
    for i in range(n):
        xh = x.copy() 
        xh[i] += h
        g1 = grad_L(fun, xh, h=h/10)
        H[:,i] = (g1 - g0)/h
    return 0.5*(H + H.T) # Makes sure the Hessian maintains required symmetry for Newton's method

def optimizer(x0,
              max_iter=50,
              rho_init=10.0,
              beta_init=1e-3,
              nu=2,
              tol=1e-6):
    x   = x0.copy()
    mu  = np.zeros(10)
    rho = rho_init # penalty parameter

    def Lfun(y):
        return aug_lagrangian(y, mu, rho)
    
    for k in range(max_iter):
        
        # Making sure the problem converges by using the Levenberg-Marquardt method
        beta = beta_init
        for i in range(100):
            g = grad_L(Lfun, x)
            H = hess_L(Lfun, x)
            try:
                p = -np.linalg.solve(H + beta*np.eye(2), g)
            except np.linalg.LinAlgError:
                beta *= nu
                continue

            if np.linalg.norm(p) < tol*(1 + np.linalg.norm(x)):
                break

            x_new = x + p
            # clip physical bounds
            x_new[0] = np.clip(x_new[0],  0.5, 10.0)
            x_new[1] = np.clip(x_new[1],  0.5, 10.0)

            f0 = Lfun(x)
            f1 = Lfun(x_new)
            pred = 0.5*p.dot(beta*p - g)
            act  = f0 - f1
            rho_ratio = act/(pred if pred>0 else 1e-16)

            if rho_ratio > 0:
                # Accept step
                x    = x_new
                beta = beta * max(1/3, 1-(2*rho_ratio-1)**3)
                nu   = 2
            else:
                # Reject step
                # Increase damping factor (beta)
                beta *= nu
                nu   *= 2

        # update the Lagrange multipliers
        gvals = constraints(x)
        mu    = np.maximum(0, mu + rho*gvals)

        # increase penalty if needed
        max_violation = np.max(np.maximum(0, gvals))
        if max_violation > 1e-3:
            rho *= 10 # increase penalty parameter

        # convergence check on original objective
        grad_norm = np.linalg.norm(grad_L(objective, x), np.inf)
        print(f"Iter {k}: x = {x[:3]}, Lagrangian = {Lfun(x):.6f}, grad norm = {grad_norm:.3e}, max violation = {max_violation:.3e}")
        if grad_norm < tol and max_violation < tol:
            break

    return x, mu

# --- usage
x0 = np.array([5.0,5.0])
sol, mu_opt = optimizer(x0)
print("Solution ct,cr,beff:", sol, "\n   Multipliers:", mu_opt)
