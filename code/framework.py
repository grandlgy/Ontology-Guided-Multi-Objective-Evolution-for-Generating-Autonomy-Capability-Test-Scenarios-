"""
LSS-CO + NSGA-II 无人机自主能力测试场景生成方法实现 (v2)
"""

import numpy as np
import xgboost as xgb
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SCENARIO_DIMENSIONS = [
    ('env_wind_speed',      'continuous', 0.0,  15.0),
    ('env_visibility',      'continuous', 50.0, 5000.0),
    ('env_illumination',    'integer',    0,    3),
    ('env_terrain',         'integer',    0,    3),
    ('env_em_interference', 'continuous', 0.0,  1.0),
    ('env_precipitation',   'continuous', 0.0,  20.0),
    ('tgt_platform_type',   'integer',    0,    1),
    ('tgt_rcs_dbsm',        'continuous', -25.0, 0.0),
    ('tgt_obstacle_density','continuous', 0.0,  1.0),
    ('mis_type',            'integer',    0,    3),
    ('mis_altitude',        'continuous', 20.0, 200.0),
    ('mis_velocity',        'continuous', 0.0,  20.0),
    ('dist_gps_loss',       'continuous', 0.0,  1.0),
    ('dist_comm_jitter',    'continuous', 0.0,  500.0),
    ('dist_payload_change', 'continuous', 0.0,  0.5),
    ('dist_battery_level',  'continuous', 10.0, 100.0),
    ('scn_init_distance',   'continuous', 50.0, 1500.0),
]

N_DIMS = len(SCENARIO_DIMENSIONS)
LB = np.array([d[2] for d in SCENARIO_DIMENSIONS], dtype=float)
UB = np.array([d[3] for d in SCENARIO_DIMENSIONS], dtype=float)


def physics_violation(x):
    v = 0.0
    wind, vis, illum, terr, em, precip = x[0], x[1], int(x[2]), int(x[3]), x[4], x[5]
    plat, rcs, obs = int(x[6]), x[7], x[8]
    mis, alt, vel = int(x[9]), x[10], x[11]
    gps, comm, payload, bat = x[12], x[13], x[14], x[15]
    if plat == 1 and mis == 1:
        v += 1.0
    if plat == 1 and vel < 5.0:
        v += (5.0 - vel) / 5.0
    if illum == 0 and vis > 2000:
        v += (vis - 2000) / 2000
    if precip > 5 and wind < 3:
        v += (3 - wind) / 3
    if em > 0.8 and gps < 0.3:
        v += (0.3 - gps) / 0.3
    if terr in (2, 3) and alt < 50 and precip > 10:
        v += (precip - 10) / 10
    return v


def repair(x):
    x = x.copy().astype(float)
    for i, d in enumerate(SCENARIO_DIMENSIONS):
        if d[1] == 'integer':
            x[i] = round(x[i])
    plat = int(x[6])
    mis  = int(x[9])
    if plat == 1 and mis == 1:
        x[9] = 0
    if plat == 1 and x[11] < 5.0:
        x[11] = 5.0
    x = np.clip(x, LB, UB)
    return x


def synthesize_failure_score(X):
    n = X.shape[0]
    score = np.zeros(n)
    for i in range(n):
        x = X[i]
        wind, vis, illum, terr, em, precip = x[0], x[1], int(x[2]), int(x[3]), x[4], x[5]
        plat, rcs, obs = int(x[6]), x[7], x[8]
        mis, alt, vel = int(x[9]), x[10], x[11]
        gps, comm, payload, bat = x[12], x[13], x[14], x[15]
        s = 0.0
        s += 0.08 * (wind / 15.0) ** 2
        if illum <= 1:
            s += 0.15 * (1.0 - min(vis, 2000) / 2000.0)
        s += 0.22 * gps
        s += 0.12 * em
        s += 0.12 * obs
        if bat < 30:
            s += 0.18 * (30 - bat) / 30
        s += 0.06 * min(comm / 500, 1.0)
        if mis == 2: s += 0.06
        if mis == 3: s += 0.10
        if terr in (1, 2):
            s += 0.05
            if alt < 50:
                s += 0.06
        if vel > 15:
            s += 0.05 * (vel - 15) / 5
        if plat == 1 and mis == 1:
            s += 0.40
        score[i] = np.clip(s, 0.0, 0.99)
    return score


def train_surrogate(n_train=8000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(LB, UB, size=(n_train, N_DIMS))
    X = np.array([repair(x) for x in X])
    y = synthesize_failure_score(X)
    y_noisy = np.clip(y + rng.normal(0, 0.03, n_train), 0.0, 1.0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_noisy, test_size=0.2, random_state=seed)
    reg = xgb.XGBRegressor(
        n_estimators=400, max_depth=7, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, verbosity=0
    )
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    return reg, rmse, r2


def predict_pfail(reg, X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.clip(reg.predict(X), 0.0, 1.0)


def coverage_score(X, n_bins=4):
    X_norm = (X - LB) / (UB - LB + 1e-12)
    X_norm = np.clip(X_norm, 0, 1 - 1e-9)
    grid = np.floor(X_norm * n_bins).astype(int)
    groups = [(0,1,2), (3,4,5), (6,7,8), (9,10,11), (12,13,14), (15,16)]
    total, hit = 0, 0
    for g in groups:
        sub = grid[:, list(g)]
        uniq = np.unique(sub, axis=0)
        n_possible = n_bins ** len(g)
        total += n_possible
        hit += min(len(uniq), n_possible)
    return hit / total


class LSSCO_Problem(Problem):
    """3目标问题: (1)最大化失效概率 (2)最小化物理违例 (3)最大化novelty(多样性)"""
    def __init__(self, surrogate, use_ontology=True):
        super().__init__(n_var=N_DIMS, n_obj=3, n_constr=0, xl=LB, xu=UB)
        self.surrogate = surrogate
        self.use_ontology = use_ontology

    def _evaluate(self, X, out, *args, **kwargs):
        # 修复+整数化
        X_eval = np.empty_like(X)
        for i in range(X.shape[0]):
            if self.use_ontology:
                X_eval[i] = repair(X[i])
            else:
                xx = X[i].copy()
                for j, d in enumerate(SCENARIO_DIMENSIONS):
                    if d[1] == 'integer':
                        xx[j] = round(xx[j])
                X_eval[i] = np.clip(xx, LB, UB)
        pfail = predict_pfail(self.surrogate, X_eval)
        f1 = -pfail  # 最大化 pfail
        # 物理违例(批量)
        if self.use_ontology:
            f2 = np.array([physics_violation(x) for x in X_eval])
        else:
            f2 = np.zeros(X.shape[0])
        # Novelty: 每个个体到种群其他个体的最近邻距离(归一化空间)
        X_norm = (X_eval - LB) / (UB - LB + 1e-12)
        # 计算距离矩阵
        diff = X_norm[:, None, :] - X_norm[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        knn = np.partition(dist, kth=min(3, dist.shape[1]-1), axis=1)[:, :3]
        novelty = knn.mean(axis=1)
        f3 = -novelty  # 最大化novelty(转为最小化)
        out["F"] = np.column_stack([f1, f2, f3])


def run_nsga2(surrogate, use_ontology, pop_size=120, n_gen=200, seed=42):
    problem = LSSCO_Problem(surrogate, use_ontology=use_ontology)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(prob=1.0/N_DIMS, eta=20),
        eliminate_duplicates=True,
    )
    res = minimize(
        problem, algorithm,
        get_termination("n_gen", n_gen),
        seed=seed, verbose=False, save_history=True,
    )
    hist_pfail, hist_cov = [], []
    for entry in res.history:
        Xs = entry.pop.get("X")
        if Xs is None or len(Xs) == 0:
            continue
        Xs_eval = np.array([repair(x) if use_ontology else x for x in Xs])
        for i, d in enumerate(SCENARIO_DIMENSIONS):
            if d[1] == 'integer':
                Xs_eval[:, i] = np.round(Xs_eval[:, i])
        Xs_eval = np.clip(Xs_eval, LB, UB)
        ps = predict_pfail(surrogate, Xs_eval)
        hist_pfail.append(float(np.mean(ps)))
        hist_cov.append(coverage_score(Xs_eval))
    X_final = res.pop.get("X")
    return X_final, hist_pfail, hist_cov


def baseline_random(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(LB, UB, size=(n, N_DIMS))
    for i, d in enumerate(SCENARIO_DIMENSIONS):
        if d[1] == 'integer':
            X[:, i] = np.round(X[:, i])
    return np.clip(X, LB, UB)


def baseline_combinatorial(n_target, seed):
    rng = np.random.default_rng(seed)
    levels = [np.linspace(LB[i], UB[i], 3) for i in range(N_DIMS)]
    pairs = []
    for i in range(N_DIMS):
        for j in range(i+1, N_DIMS):
            for a in levels[i]:
                for b in levels[j]:
                    pairs.append((i, j, a, b))
    rng.shuffle(pairs)
    samples = []
    while len(samples) < n_target and pairs:
        i, j, a, b = pairs.pop()
        x = np.array([levels[k][rng.integers(0,3)] for k in range(N_DIMS)])
        x[i] = a; x[j] = b
        samples.append(x)
    while len(samples) < n_target:
        samples.append(np.array([levels[k][rng.integers(0,3)] for k in range(N_DIMS)]))
    X = np.array(samples[:n_target])
    for i, d in enumerate(SCENARIO_DIMENSIONS):
        if d[1] == 'integer':
            X[:, i] = np.round(X[:, i])
    return np.clip(X, LB, UB)


def evaluate_set(X, surrogate, ontology_used=True):
    if ontology_used:
        X_eval = np.array([repair(x) for x in X])
    else:
        X_eval = X.copy().astype(float)
        for i, d in enumerate(SCENARIO_DIMENSIONS):
            if d[1] == 'integer':
                X_eval[:, i] = np.round(X_eval[:, i])
        X_eval = np.clip(X_eval, LB, UB)
    pfail = predict_pfail(surrogate, X_eval)
    cov = coverage_score(X_eval)
    crit = float((pfail > 0.5).mean())
    viol = float(np.mean([physics_violation(x) > 0.01 for x in X_eval]))
    return {
        'coverage': cov,
        'crit_ratio': crit,
        'phys_viol_ratio': viol,
        'mean_pfail': float(pfail.mean()),
        'n_scenarios': len(X_eval),
    }
