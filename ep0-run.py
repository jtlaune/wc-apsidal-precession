import numpy as npy
import scipy as sp
from loky import get_reusable_executor
import dill as pickle

TMVAL = 2 * npy.pi * 1e6


def solve_ivp(job):
    ind, e_p, omega_p, omega = job[0], job[1], job[2], job[3]

    funcl = lambda n, e, varpi, theta_p, n_p, t: (
        [
            0.000853042833571296
            * e
            * n ** (4 / 3)
            * n_p ** (2 / 3)
            * npy.sin(omega_p * t + theta_p - varpi)
            - 0.000695490377254464
            * e_p
            * n ** (4 / 3)
            * n_p ** (2 / 3)
            * npy.sin(theta_p),
            0.000142173805595216
            * e**2
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.sin(omega_p * t + theta_p - varpi)
            - 0.000115915062875744
            * e
            * e_p
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.sin(theta_p)
            - 9.47825370634774e-5
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.sin(omega_p * t + theta_p - varpi),
            omega
            + 9.47825370634774e-5
            * n ** (1 / 3)
            * n_p ** (2 / 3)
            * npy.cos(omega_p * t + theta_p - varpi)
            / e,
            0.000142173805595216
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
        if t > 2 * npy.pi * 3e5:
            npval = npy.exp(-2 * npy.pi * 3e5 / TMVAL)
        else:
            npval = npy.exp(-t / TMVAL)
        return funcl(*x0, npval, t)

    teval = npy.linspace(0, 2 * npy.pi * 6e5, 1000000)
    sol = sp.integrate.solve_ivp(
        func,
        [teval[0], teval[-1]],
        [1 / 1.6, 0.001, 0, 0],
        t_eval=teval,
        rtol=1e-9,
        atol=1e-9,
        method="DOP853",
    )

    print(ind)
    return sol


alpha0val = (2 / 3) ** (2.0 / 3)
# epvals = npy.linspace(5e-3,0.1,20)
# ompvals = npy.zeros(31)
# ompvals[1:] = npy.logspace(-5, -2.7865, 30)
# jobs = []
# for i, epval in enumerate(epvals):
#    for k, ompval in enumerate(ompvals):
#        jobs = jobs + [(len(ompvals) * i + k, epval, ompval, ompval * alpha0val**3.5)]
#
# selects = [119,105,305,291,491,477,12, 198, 384, 570,26, 212, 398, 584,5, 98, 191, 284, 377, 470, 563,19, 112, 205, 298, 391, 484, 577, 279]
# jobs = [jobs[select] for select in selects]

jobs = [
    (0, 0, 0, 0 * alpha0val**3.5),
    (0, 0, 1e-1, 1e-1 * alpha0val**3.5),
]


executors_solve_ivp = get_reusable_executor(max_workers=len(jobs))
results = list(executors_solve_ivp.map(solve_ivp, jobs))
with open("ep0_mup5e-5_tm2pi1e6_n1.6_omalpha0val3.5.pkl", "wb") as f:
    pickle.dump([jobs, results], f)
