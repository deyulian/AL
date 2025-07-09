import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# === Data Parser (for CSV, included for completeness) ===
class DataParser:
    def __init__(self, data_str):
        self.data_str = data_str

    def parse(self):
        buf = StringIO(self.data_str)
        cols = buf.readline().strip().split(',')
        dt = np.genfromtxt(buf, delimiter=',')
        y = dt[:,0]
        X = dt[:,1:]
        return X, y, cols[1:]

# === Test Functions ===
def forrester(x):
    x = np.array(x)
    return (6*x - 2)**2 * np.sin(12*x - 4)

def rosenbrock5(x):
    x = np.array(x)
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2)

def sobol_g(x, a=None):
    x = np.array(x)
    if a is None:
        a = np.ones(x.size)
    return np.prod((np.abs(4*x-2)+a)/(1+a))

# === Utilities ===
def featurize(X):
    return np.hstack([X, np.sin(2*np.pi*X), np.cos(2*np.pi*X)])

def simulate_samples(X, func):
    y = np.array([func(x) for x in X]).reshape(-1,1)
    return X, y

def surrogate_model(X, y, rf_params):
    Xf = featurize(X)
    scaler = StandardScaler().fit(Xf)
    Xs = scaler.transform(Xf)
    model = RandomForestRegressor(oob_score=True, **rf_params)
    model.fit(Xs, y.ravel())
    return model, scaler

def evaluate_model(model, scaler, X, y):
    Xs = scaler.transform(featurize(X))
    preds = model.predict(Xs)
    return r2_score(y, preds), math.sqrt(mean_squared_error(y, preds))

def samples_from_mixture(X_existing, weights, means, sigma, N):
    samples = []
    existing = set(map(tuple, X_existing))
    dim = means.shape[1]
    while len(samples) < N:
        idx = np.random.choice(len(weights), p=weights)
        s = np.random.multivariate_normal(means[idx], np.eye(dim)*(sigma[idx]**2))
        if tuple(s) not in existing:
            samples.append(s)
            existing.add(tuple(s))
    return np.array(samples)

# === Adaptive Sampler ===
def adaptive_sampler(func, bounds, X_dim, init_N, iter_N, max_samples,
                     n_iter, val_N, rf_params,
                     err_frac_max=0.5, pool_factor=20, exploration_iters=3):
    lo, hi = bounds
    x_min, x_max = np.full(X_dim, lo), np.full(X_dim, hi)
    # validation set
    X_val = x_min + np.random.rand(val_N, X_dim)*(x_max - x_min)
    X_val, y_val = simulate_samples(X_val, func)
    # initial design
    X_full = x_min + np.random.rand(init_N, X_dim)*(x_max - x_min)
    X_full = np.unique(X_full, axis=0)
    X_full, y_full = simulate_samples(X_full, func)
    # records
    sample_counts, r2_tr, r2_vl, rm_vl = [], [], [], []

    for i in range(n_iter+1):
        model, scaler = surrogate_model(X_full, y_full, rf_params)
        rt, _ = evaluate_model(model, scaler, X_full, y_full)
        rv, rm = evaluate_model(model, scaler, X_val, y_val)
        sample_counts.append(len(X_full)); r2_tr.append(rt); r2_vl.append(rv); rm_vl.append(rm)
        if i % 5 == 0:
          print(f"Iter {i:>2}: Samples={len(X_full)}, Train R²={rt:.4f}, Val R²={rv:.4f}, Val RMSE={rm:.4f}")
        if i == n_iter:
            break

        # dynamic err_frac
        if i < exploration_iters:
            err_frac = 0.0
        else:
            err_frac = err_frac_max * ((i-exploration_iters)/(n_iter-exploration_iters))

        # residual model
        residuals = y_full.ravel() - model.oob_prediction_
        rf_resid = RandomForestRegressor(**rf_params)
        rf_resid.fit(scaler.transform(featurize(X_full)), residuals)

        # candidate pool
        pool_size = iter_N * pool_factor
        X_pool = x_min + np.random.rand(pool_size, X_dim)*(x_max - x_min)
        Xs_pool = scaler.transform(featurize(X_pool))

        # scores
        err_scores = np.abs(rf_resid.predict(Xs_pool))
        all_preds = np.stack([t.predict(Xs_pool) for t in model.estimators_], axis=0)
        unc_scores = all_preds.std(axis=0)

        # select
        n_err = int(iter_N * err_frac); n_unc = iter_N - n_err
        idx_err = np.argsort(-err_scores)[:n_err]
        idx_unc = []
        for idx in np.argsort(-unc_scores):
            if idx not in idx_err and len(idx_unc) < n_unc:
                idx_unc.append(idx)
            if len(idx_unc) >= n_unc:
                break
        idx_sel = np.concatenate([idx_err, idx_unc])
        X_new = X_pool[idx_sel]

        # append new
        Xn, yn = simulate_samples(X_new, func)
        X_full = np.vstack([X_full, Xn]); y_full = np.vstack([y_full, yn])
        # cap
        if len(X_full) > max_samples:
            X_full = X_full[-max_samples:]; y_full = y_full[-max_samples:]

    return np.array(sample_counts), np.array(r2_tr), np.array(r2_vl), np.array(rm_vl)

# === Random Sampler ===
def random_sampler(func, bounds, X_dim, sample_counts, val_N, rf_params):
    lo, hi = bounds
    x_min, x_max = np.full(X_dim, lo), np.full(X_dim, hi)
    X_val = x_min + np.random.rand(val_N, X_dim)*(x_max - x_min)
    X_val, y_val = simulate_samples(X_val, func)
    r2_tr, r2_vl, rm_vl = [], [], []
    for i, n in enumerate(sample_counts):
        X_tr = x_min + np.random.rand(int(n), X_dim)*(x_max - x_min)
        _, y_tr = simulate_samples(X_tr, func)
        model, scaler = surrogate_model(X_tr, y_tr, rf_params)
        rt, _ = evaluate_model(model, scaler, X_tr, y_tr)
        rv, rm = evaluate_model(model, scaler, X_val, y_val)
        r2_tr.append(rt); r2_vl.append(rv); rm_vl.append(rm)
        if i % 5 == 0:
          print(f"Rand Iter {i:>2}: Samples={n}, Val R²={rv:.4f}, RMSE={rm:.4f}")
    return np.array(r2_tr), np.array(r2_vl), np.array(rm_vl)

# === Main Execution & Plot ===
if __name__ == "__main__":
    experiments = [
        ("Forrester (1D)", forrester, (0.0, 1.0), 1, 100, 30, 3000, 0.5, 100, 100),
        ("Rosenbrock (5D)", rosenbrock5, (-2.0, 2.0), 5, 100, 30, 3000, 0.5, 100, 250),
        ("Sobol g (10D)", sobol_g, (0.0, 1.0), 10, 100, 30, 3000, 0.5, 100, 500),
    ]
    rf_params = {"n_estimators":100, "max_depth":None, "random_state":0, "n_jobs":1}

    for title, func, bounds, X_dim, init_N, iter_N, max_s, err_frac, n_iter, val_N in experiments:
        print(f"\n=== {title} ===")
        sc, r2_tr_ad, r2_vl_ad, rm_ad = adaptive_sampler(
            func, bounds, X_dim, init_N, iter_N, max_s,
            n_iter, val_N, rf_params, err_frac_max=err_frac, pool_factor=20, exploration_iters=3
        )
        r2_tr_rs, r2_vl_rs, rm_rs = random_sampler(
            func, bounds, X_dim, sc, val_N, rf_params
        )

        # Adaptive plot
        fig, ax1 = plt.subplots(figsize=(6,3))
        ax1.plot(sc, r2_tr_ad, 'o-', label='Adaptive Train R²')
        ax1.plot(sc, r2_vl_ad, 's-', label='Adaptive Val R²')
        ax1.set_xlabel('Samples'); ax1.set_ylabel('R²'); ax1.set_ylim(-0.1,1.05)
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(sc, rm_ad, '^-', label='Adaptive Val RMSE'); ax2.set_ylabel('RMSE')
        ax2.legend(loc='upper right')
        plt.title(f'{title} - Adaptive'); plt.tight_layout(); plt.show()

        # Random plot
        fig, ax1 = plt.subplots(figsize=(6,3))
        ax1.plot(sc, r2_tr_rs, 'x--', label='Random Train R²')
        ax1.plot(sc, r2_vl_rs, 'd--', label='Random Val R²')
        ax1.set_xlabel('Samples'); ax1.set_ylabel('R²'); ax1.set_ylim(-0.1,1.05)
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.plot(sc, rm_rs, 'v--', label='Random Val RMSE'); ax2.set_ylabel('RMSE')
        ax2.legend(loc='upper right')
        plt.title(f'{title} - Random'); plt.tight_layout(); plt.show()
