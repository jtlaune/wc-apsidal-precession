import LaplaceCoefficients
import numpy as npy
import scipy as sp
from loky import get_reusable_executor
import dill as pickle

# TODO: really need to figure out a better way to do this + the lambdastr

PARAMS = {
    "h": 0.1,
    "mu1": 3e-6,
    "mu2": 6e-6,
    "Tm2": 1e7 * 2 * npy.pi,
}
PARAMS["q"] = PARAMS["mu1"] / PARAMS["mu2"]
PARAMS["Te20"] = PARAMS["Tm2"] * PARAMS["h"] ** 2
PARAMS["Te10"] = PARAMS["Te20"] / PARAMS["q"]

SEED = 3330
N_JOBS = 1000
BATCH_SIZE = 100
if N_JOBS % BATCH_SIZE != 0:
    raise Warning("BATCH_SIZE must divide N_JOBS.")


def solve_ivp(job):
    ind, GPa3, GPe3, theta0, varpi10, varpi20 = (
        job[0],
        job[1],
        job[2],
        job[3],
        job[4],
        job[5],
    )
    funcl = lambda _Dummy_73, _Dummy_74, _Dummy_75, _Dummy_76, _Dummy_77, _Dummy_78, _Dummy_79, _Dummy_80, _Dummy_81, _Dummy_82, _Dummy_83, _Dummy_84, _Dummy_85, _Dummy_86, _Dummy_87: (
        [
            5.56392301803571e-5
            * _Dummy_73 ** (4 / 3)
            * _Dummy_74 ** (2 / 3)
            * _Dummy_75
            * npy.sin(_Dummy_77 - _Dummy_78)
            - 6.82434266857037e-5
            * _Dummy_73 ** (4 / 3)
            * _Dummy_74 ** (2 / 3)
            * _Dummy_76
            * npy.sin(_Dummy_77 - _Dummy_79)
            + 3 * _Dummy_75**2 * _Dummy_73 / _Dummy_86,
            -4.17294226352678e-5
            * _Dummy_74**2
            * _Dummy_75
            * npy.sin(_Dummy_77 - _Dummy_78)
            + 5.11825700142778e-5
            * _Dummy_74**2
            * _Dummy_76
            * npy.sin(_Dummy_77 - _Dummy_79)
            + 3 * _Dummy_76**2 * _Dummy_74 / _Dummy_87
            + 2.38732414637843e-8 * _Dummy_74,
            9.27320503005952e-6
            * _Dummy_73 ** (1 / 3)
            * _Dummy_74 ** (2 / 3)
            * _Dummy_75**2
            * npy.sin(_Dummy_77 - _Dummy_78)
            - 1.13739044476173e-5
            * _Dummy_73 ** (1 / 3)
            * _Dummy_74 ** (2 / 3)
            * _Dummy_75
            * _Dummy_76
            * npy.sin(_Dummy_77 - _Dummy_79)
            + 9.27320503005952e-6
            * _Dummy_73 ** (1 / 3)
            * _Dummy_74 ** (2 / 3)
            * npy.sin(_Dummy_77 - _Dummy_78)
            - _Dummy_75 / _Dummy_86
            - 1.57285601129232e-5
            * _Dummy_74 ** (4 / 3)
            * _Dummy_76
            * npy.sin(_Dummy_78 - _Dummy_79)
            / _Dummy_73 ** (1 / 3)
            - 0.00025
            * _Dummy_81
            * _Dummy_83
            * npy.sin(_Dummy_78)
            / (_Dummy_73 ** (1 / 3) * _Dummy_80**2),
            -6.95490377254464e-6
            * _Dummy_74
            * _Dummy_75
            * _Dummy_76
            * npy.sin(_Dummy_77 - _Dummy_78)
            + 8.53042833571296e-6
            * _Dummy_76**2
            * _Dummy_74
            * npy.sin(_Dummy_77 - _Dummy_79)
            - 5.68695222380864e-6 * _Dummy_74 * npy.sin(_Dummy_77 - _Dummy_79)
            - _Dummy_76 / _Dummy_87
            - 0.00025
            * _Dummy_81
            * _Dummy_85
            * npy.sin(_Dummy_79)
            / (_Dummy_74 ** (1 / 3) * _Dummy_80**2)
            + 7.86428005646161e-6
            * _Dummy_74 ** (5 / 3)
            * _Dummy_75
            * npy.sin(_Dummy_78 - _Dummy_79)
            / _Dummy_73 ** (2 / 3),
            9.27320503005952e-6
            * _Dummy_73 ** (1 / 3)
            * _Dummy_74 ** (2 / 3)
            * _Dummy_75
            * npy.cos(_Dummy_77 - _Dummy_78)
            - 2 * _Dummy_73
            + 8.53042833571296e-6
            * _Dummy_74
            * _Dummy_76
            * npy.cos(_Dummy_77 - _Dummy_79)
            + 3 * _Dummy_74,
            -9.27320503005952e-6
            * _Dummy_73 ** (1 / 3)
            * _Dummy_74 ** (2 / 3)
            * npy.cos(_Dummy_77 - _Dummy_78)
            / _Dummy_75
            + 1.81271409306947e-5 * _Dummy_74 ** (4 / 3) / _Dummy_73 ** (1 / 3)
            - 1.57285601129232e-5
            * _Dummy_74 ** (4 / 3)
            * _Dummy_76
            * npy.cos(_Dummy_78 - _Dummy_79)
            / (_Dummy_73 ** (1 / 3) * _Dummy_75)
            + 0.00025 * _Dummy_82 / (_Dummy_73 ** (1 / 3) * _Dummy_80**2)
            - 0.00025
            * _Dummy_81
            * _Dummy_83
            * npy.cos(_Dummy_78)
            / (_Dummy_73 ** (1 / 3) * _Dummy_75 * _Dummy_80**2),
            5.68695222380864e-6 * _Dummy_74 * npy.cos(_Dummy_77 - _Dummy_79) / _Dummy_76
            + 0.00025 * _Dummy_84 / (_Dummy_74 ** (1 / 3) * _Dummy_80**2)
            - 0.00025
            * _Dummy_81
            * _Dummy_85
            * npy.cos(_Dummy_79)
            / (_Dummy_74 ** (1 / 3) * _Dummy_76 * _Dummy_80**2)
            - 7.86428005646161e-6
            * _Dummy_74 ** (5 / 3)
            * _Dummy_75
            * npy.cos(_Dummy_78 - _Dummy_79)
            / (_Dummy_73 ** (2 / 3) * _Dummy_76)
            + 9.06357046534736e-6 * _Dummy_74 ** (5 / 3) / _Dummy_73 ** (2 / 3),
        ]
    )

    def func(t, x0):
        # x0=[n1, n2, e1, e2, th0, pom1, pom2]

        VALe1 = x0[2]
        VALe2 = x0[3]

        VALte1 = PARAMS["Te10"] * (
            1 - 0.14 * (VALe1 / PARAMS["h"]) ** 2 + 0.06 * (VALe1 / PARAMS["h"]) ** 3
        )
        VALte2 = PARAMS["Te20"] * (
            1 - 0.14 * (VALe2 / PARAMS["h"]) ** 2 + 0.06 * (VALe2 / PARAMS["h"]) ** 3
        )

        VALa1 = x0[0] ** (-2.0 / 3)
        VALa2 = x0[1] ** (-2.0 / 3)

        VALb1_3_2_13 = LaplaceCoefficients.b(1.5, 1, VALa1 / GPa3)
        VALb2_3_2_13 = LaplaceCoefficients.b(1.5, 2, VALa1 / GPa3)
        VALb1_3_2_23 = LaplaceCoefficients.b(1.5, 1, VALa2 / GPa3)
        VALb2_3_2_23 = LaplaceCoefficients.b(1.5, 2, VALa2 / GPa3)

        return funcl(
            *x0,
            GPa3,
            GPe3,
            VALb1_3_2_13,
            VALb2_3_2_13,
            VALb1_3_2_23,
            VALb2_3_2_23,
            VALte1,
            VALte2,
        )

    def event1(t, Y):
        return Y[1] / Y[0] - 0.7

    event1.terminal = True

    teval = npy.linspace(0, 2 * npy.pi * 5e5, 10000)
    sol = sp.integrate.solve_ivp(
        func,
        [teval[0], teval[-1]],
        [1.57, 1, 0.001, 0.001, theta0, varpi10, varpi20],
        t_eval=teval,
        rtol=1e-9,
        atol=1e-9,
        method="DOP853",
        events=[event1],
    )
    if ind % 100 == 0:
        print(f"{ind}...")
    return sol


a3s = npy.random.default_rng(seed=SEED).uniform(2, 7, N_JOBS)
e3s = npy.random.default_rng(seed=SEED).uniform(0.0, 0.3, N_JOBS)
theta0s = npy.random.default_rng(seed=SEED).uniform(0.0, 2 * npy.pi, N_JOBS)
varpi10s = npy.random.default_rng(seed=SEED).uniform(0.0, 2 * npy.pi, N_JOBS)
varpi20s = npy.random.default_rng(seed=SEED).uniform(0.0, 2 * npy.pi, N_JOBS)

# npy.power(
#    10, npy.random.default_rng(seed=SEED).uniform(npy.log10(5e-6), -3, N_JOBS)
# )

jobs = []

for i in range(N_JOBS):
    jobs += [
        (
            i,
            a3s[i],
            e3s[i],
            theta0s[i],
            varpi10s[i],
            varpi20s[i],
        )
    ]

README = """mu3 = 1e-3
pom3 = 0
Tm2 = 2pi*1e7
h=0.1
Realistic eccentricity damping
Initial n1/n2 = 1.57"""

executors_solve_ivp = get_reusable_executor(max_workers=60)
results = []
for ibatch in range(int(N_JOBS / BATCH_SIZE)):
    results = results + list(
        executors_solve_ivp.map(
            solve_ivp, jobs[ibatch * BATCH_SIZE : (ibatch + 1) * BATCH_SIZE]
        )
    )
    print(f"Saving i={(ibatch+1)*BATCH_SIZE}")
    with open(f"results_1ME_2ME_random_grid_{N_JOBS}.pkl", "wb") as f:
        f.write(README.encode()  + b"\n---\n")
        pickle.dump([jobs, results], f)
