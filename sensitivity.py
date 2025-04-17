import numpy as np
import matplotlib.pyplot as plt

def objective(x):
    ct, cr, beff = x
    return (ct + cr) / (2 * beff)

def aug_lagrangian(x, mu, rho, constraints):
    f0 = objective(x)
    g = constraints(x)
    penalty = np.sum(np.maximum(0, g)**2)
    return f0 + np.dot(mu, g) + 0.5 * rho * penalty

def grad_L(Lfun, x, h=1e-8):
    n = len(x)
    f0 = Lfun(x)
    grad = np.zeros(n)
    for i in range(n):
        xh = x.copy()
        xh[i] += h
        grad[i] = (Lfun(xh) - f0) / h
    return grad

def hess_L(fun, x, h=1e-5):
    n = len(x)
    H = np.zeros((n,n))
    g0 = grad_L(fun, x, h=h/10)
    for i in range(n):
        xh = x.copy()
        xh[i] += h
        g1 = grad_L(fun, xh, h=h/10)
        H[:,i] = (g1 - g0) / h
    return 0.5 * (H + H.T)

def make_constraints(params):
    taper_min = params['taper_min']
    taper_max = params['taper_max']
    AR_min = params['AR_min']
    AR_max = params['AR_max']
    chord_min = params['chord_min']
    chord_max = params['chord_max']
    area_min = params['area_min']
    area_max = params['area_max']
    beff_min = params['beff_min']
    beff_max = params['beff_max']
    
    def constraints(x):
        ct, cr, beff = x
        S = ct + cr
        taper = ct / cr
        halfchord = 30 * np.pi / 180
        sqrt_t = np.sqrt(1 - np.sin(halfchord)**2)
        return np.array([
            taper - taper_max,
            -taper + taper_min,
            2*beff/S - AR_max,
            -2*beff/S + AR_min,
            (2/3)*cr*((1+taper+taper**2)/(1+taper)) - chord_max,
            -((2/3)*cr*((1+taper+taper**2)/(1+taper))) + chord_min,
            beff*S/(2*sqrt_t) - area_max,
            -beff*S/(2*sqrt_t) + area_min,
            beff - beff_max,
            -beff + beff_min,
            -ct,
            -cr
        ])
    return constraints

def optimizer(params, x0=np.array([5.0,5.0,50.0]), max_iter=50, rho_init=10.0, beta_init=1e-3, nu=2, tol=1e-6):
    constraints = make_constraints(params)
    x = x0.copy()
    mu = np.zeros(12)
    rho = rho_init
    
    def Lfun(y):
        return aug_lagrangian(y, mu, rho, constraints)
    
    for k in range(max_iter):
        beta = beta_init
        for _ in range(100):
            g = grad_L(Lfun, x)
            H = hess_L(Lfun, x)
            try:
                p = -np.linalg.solve(H + beta*np.eye(3), g)
            except np.linalg.LinAlgError:
                beta *= nu
                continue
            if np.linalg.norm(p) < tol*(1 + np.linalg.norm(x)):
                break
            x_new = x + p
            x_new[0] = np.clip(x_new[0], params['ct_min'], params['ct_max'])
            x_new[1] = np.clip(x_new[1], params['cr_min'], params['cr_max'])
            x_new[2] = np.clip(x_new[2], params['beff_min'], params['beff_max'])
            f0, f1 = Lfun(x), Lfun(x_new)
            pred = 0.5 * p.dot(beta*p - g)
            act  = f0 - f1
            rho_ratio = act / (pred if pred > 0 else 1e-16)
            if rho_ratio > 0:
                x = x_new
                beta = beta * max(1/3, 1 - (2*rho_ratio-1)**3)
                nu   = 2
            else:
                beta *= nu
                nu   *= 2
        gvals = constraints(x)
        mu = np.maximum(0, mu + rho*gvals)
        max_violation = np.max(np.maximum(0, gvals))
        if max_violation > 1e-3:
            rho *= 10
        grad_norm = np.linalg.norm(grad_L(objective, x), np.inf)
        if grad_norm < tol and max_violation < tol:
            break
    return x, objective(x)

baseline = {
    'taper_min': 0.21, 'taper_max': 0.31,
    'AR_min': 7.5, 'AR_max': 9.5,
    'chord_min': 4, 'chord_max': 12,
    'area_min': 120, 'area_max': 820,
    'beff_min': 33, 'beff_max': 80,
    'ct_min': 0.5, 'ct_max': 10.0,
    'cr_min': 0.5, 'cr_max': 10.0
}

# Parameters to vary
vary_params = ['taper_max', 'AR_max', 'area_max']
ranges = {
    'taper_max': np.linspace(0.8*baseline['taper_max'], 1.2*baseline['taper_max'], 11),
    'AR_max':    np.linspace(0.8*baseline['AR_max'],    1.2*baseline['AR_max'],    11),
    'area_max':  np.linspace(0.8*baseline['area_max'],  1.2*baseline['area_max'],  11)
}

results = {param: {'values': [], 'ct': [], 'cr': [], 'beff': [], 'obj': []} for param in vary_params}

# Run sensitivity analysis
for param in vary_params:
    for val in ranges[param]:
        params = baseline.copy()
        params[param] = val
        sol, obj = optimizer(params)
        results[param]['values'].append(val)
        results[param]['ct'].append(sol[0])
        results[param]['cr'].append(sol[1])
        results[param]['beff'].append(sol[2])
        results[param]['obj'].append(obj)

# Plotting results
for param in vary_params:
    plt.figure()
    plt.plot(results[param]['values'], results[param]['ct'], label='ct')
    plt.plot(results[param]['values'], results[param]['cr'], label='cr')
    plt.plot(results[param]['values'], results[param]['beff'], label='beff')
    plt.xlabel(param)
    plt.ylabel('Design variables')
    plt.legend()
    plt.title(f"Sensitivity of (ct, cr, beff) to {param}")

    plt.figure()
    plt.plot(results[param]['values'], results[param]['obj'])
    plt.xlabel(param)
    plt.ylabel('Objective value')
    plt.title(f"Sensitivity of objective to {param}")

plt.show()
