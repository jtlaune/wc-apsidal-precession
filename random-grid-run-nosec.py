import numpy as npy
import scipy as sp
from loky import get_reusable_executor
import dill as pickle

TMVAL = 2 * npy.pi * 1e6


def solve_ivp(job):
    ind, e_p, omega_p, omega, thetap0 = (
        job[0],
        job[1],
        job[2],
        job[3],
        job[4],
    )

    funcl = lambda n, e, varpi, theta_p, n_p, t: (
        [
            0.00170608566714259
            * e
            * n ** (4 / 3)
            * n_p ** (2 / 3)
            * npy.sin(omega_p * t + theta_p - varpi)
            - 0.00139098075450893
            * e_p
            * n ** (4 / 3)
            * n_p ** (2 / 3)
            * npy.sin(theta_p),
            0.000284347611190432
            * e**2
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.sin(omega_p * t + theta_p - varpi)
            - 0.000231830125751488
            * e
            * e_p
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.sin(theta_p)
            - 0.000189565074126955
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.sin(omega_p * t + theta_p - varpi),
            omega
            + 0.000189565074126955
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.cos(omega_p * t + theta_p - varpi)
            / e,
            0.000284347611190432
            * e
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.cos(omega_p * t + theta_p - varpi)
            + 3 * n
            - 2 * n_p
            - omega_p,
        ]
    )

    def func(t, x0):
        npval = npy.exp(-t / TMVAL)
        return funcl(*x0, npval, t)

    def event1(t, Y):
        npval = npy.exp(-t / TMVAL)
        return Y[0] / npval - 0.8

    event1.terminal = True

    teval = npy.linspace(0, 2 * npy.pi * 7.5e5, 10000)
    sol = sp.integrate.solve_ivp(
        func,
        [teval[0], teval[-1]],
        # ICs: n, e, varpi, theta_p
        [1 / 1.55, 0.001, 0, thetap0],
        t_eval=teval,
        rtol=1e-9,
        ###################################################
        # Changed atol from other running scripts 10/1/24 #
        ###################################################
        atol=1e-9,
        method="DOP853",
        events=[event1],
    )

    if ind % 100 == 0:
        print(ind)
    return sol


alpha0val = (2 / 3) ** (2.0 / 3)

N_jobs = 10000

# Uniform in epvals
epvals = npy.random.default_rng(seed=182).uniform(5e-3, 0.1, N_jobs)

# Log uniform in ompvals
ompvals = npy.power(
    10, npy.random.default_rng(seed=5802).uniform(npy.log10(5e-6), -3, N_jobs)
)

# Uniform in thetap0vals
thetap0vals = npy.random.default_rng(seed=3801).uniform(0, 2 * npy.pi, N_jobs)

jobs = []

for i in range(N_jobs):
    jobs += [(i, epvals[i], ompvals[i], ompvals[i] * alpha0val**3.5, thetap0vals[i])]

executors_solve_ivp = get_reusable_executor(max_workers=60)
results = list(executors_solve_ivp.map(solve_ivp, jobs))
with open(
    f"results_mup1e-4_tm2pi1e6_n1.55_omalpha0val3.5_random{N_jobs}_nosec.pkl", "wb"
) as f:
    pickle.dump([jobs, results], f)
