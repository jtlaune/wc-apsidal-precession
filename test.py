import os
import sympy as sm
import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as ticker

from matplotlib import pyplot as plt
from mpl_styles import analytic
from scipy.special import hyp2f1
#from sympy import init_printing, init_session
from sympy import sqrt, Rational, Matrix, eye, BlockMatrix, zeros,lambdify
plt.style.use(analytic)
#init_session()
#init_printing()

def Pochhammer(a, k):
    if k == 0:
        return 1.0
    else:
        return (a + k - 1) * Pochhammer(a, k - 1)


def b(s, j, alpha):
    if j >= 0:  # Eq. 7.87
        return (
            2
            * Pochhammer(s, j)
            / Pochhammer(1, j)
            * (alpha**j)
            * hyp2f1(s, s + j, j + 1, alpha * alpha)
        )
    else:  # Eq. 6.69
        return b(s, -j, alpha)


def Db(s, j, alpha):  # Eq. 6.70
    aux = (
        b(s + 1, j - 1, alpha) - 2 * alpha * b(s + 1, j, alpha) + b(s + 1, j + 1, alpha)
    )
    return s * aux


def D2b(s, j, alpha):  # Eq. 6.71
    aux = (
        Db(s + 1, j - 1, alpha)
        - 2 * alpha * Db(s + 1, j, alpha)
        - 2 * b(s + 1, j, alpha)
        + Db(s + 1, j + 1, alpha)
    )
    return s * aux


def f27lc(alpha, j):
    """
    f27 in MD p543
    (1/2)[−2 j − αD] b^(j)_{1/2}(α) x [e1cos(theta1)]
    """
    return 0.5 * (-2 * (j + 1) * b(0.5, j + 1, alpha) - alpha * Db(0.5, j + 1, alpha))


def f31lc(alpha, j):
    """
    f31 in MD p543
    (1/2)[−1 + 2 j + αD] b^(j-1)_{1/2}(α) x [e2cos(theta2)]
    """
    return 0.5 * ((-1 + 2 * (j + 1)) * b(0.5, j, alpha) + alpha * Db(0.5, j, alpha))

alpha0 = (2 / (2 + 1)) ** (2.0 / 3)

############################################################
# If test particle is outside, have to multiply by alpha_0 #
############################################################

f1val_outside = alpha0*f27lc(alpha0, 2)
f2val_outside = alpha0*f31lc(alpha0, 2)


f1, f2, f3, f4, tau, l, pom, pom_p, g, om, om_p = sm.symbols(
    "f_1 f_2 f_3 f_4 tau lambda varpi varpi_p gamma omega omega_p"
)
a, a_p, e, e_p, n, n_p, m_p, mu_p, L, G, Gconst, M, j = sm.symbols(
    "a a_p e e_p n n_p m_p mu_p Lambda Gamma G M j", positive=True
)
X, Y, Te, Tm = sm.symbols("X Y T_e T_m")

H_scale_factor = Gconst * M / a_p
t_scale_factor = sqrt(Gconst * M / a_p**3)  # = n_p
L_scale_factor = H_scale_factor / t_scale_factor

calH = (
    -(Gconst * M) / (2 * a)
    - Gconst
    * m_p
    / a_p
    * (
        e * f2 * sm.cos((j + 1) * l - j * tau - pom)
        + e_p * f1 * sm.cos((j + 1) * l - j * tau - pom_p - om_p * tau)
    )
    - (Gconst * m_p / a_p * (f3 * e * e_p + f4 * (e**2 + e_p**2)))
    - Rational(1, 2) * sqrt(Gconst * M * a) * e**2 * om * t_scale_factor
)


calH = (
    (calH / H_scale_factor)
    .subs(m_p, mu_p * M)
    .simplify()
    .expand()
    .subs({f3: 0, f4: 0})
    #.subs(a, a * a_p)
)


def to_canonical(expr):
    return expr.subs({a: L**2, e: sqrt(2 * G / L), pom: -g})


def to_orbelts(expr):
    # This uses the approximation G=1/2*L*e^2
    return expr.subs({g: -pom, G: Rational(1, 2) * L * e**2, L: sqrt(a)})


Hcanon = to_canonical(calH)
Hcanon


x = Matrix([l,g,L,G])
x


pHpx = Matrix([Hcanon]).jacobian(x).T
pHpx


J = Matrix(BlockMatrix([[zeros(2,2),eye(2)],[-eye(2),zeros(2,2)]]))
J


dxdt = J*pHpx
dxdt


p2Hpx2 = pHpx.jacobian(x)
p2Hpx2


dxdt


y1, y2, y3, y4 = sm.symbols("y_1 y_2 y_3 y_4")
y = Matrix([y1, y2, y3, y4])
y


dydt = J*p2Hpx2*y
dydt


RHS = Matrix.vstack(dxdt, dydt)


xy = Matrix([l,g,L,G,y1,y2,y3,y4])

# # Grid of runs


params = {
    a_p:1,
    e_p:0.03,
    mu_p:1e-3,
    pom_p:0,
    om:0,
    om_p:0,
    f1:f1val_outside,
    f2:f2val_outside,
    j:2,
}
rhs = RHS.subs(params)
rhs.free_symbols
rhsfunc = lambdify([tau,*xy],rhs)
def func(t, Y):
    return rhsfunc(t, Y[0], Y[1], Y[2], Y[3], Y[4], Y[5], Y[6], Y[7])[:,0]

def calc_FLIt(p):
    i, av, th2v = p[0], p[1], p[2]
    print(i)
    initorb = {
        a:av,
        e:0.03,
        pom:0,
        l:th2v/(params[j]+1)
    }
    initcanon = {
        l:to_orbelts(l).subs(initorb),
        g:to_orbelts(g).subs(initorb),
        L:to_orbelts(L).subs(initorb),
        G:to_orbelts(G).subs(initorb),
        tau:0,
    }
    inity = [0,0,0,1]
    inity = {y[i]:el for i, el in enumerate(inity)}

    Y0 = np.float64(xy.subs(initcanon).subs(inity))[:,0]
    t_span = (0,1e4)
    sol = sp.integrate.solve_ivp(func, t_span, Y0)
    dx_t = sol.y[4:,:]
    return(np.log(np.linalg.norm(dx_t[:,:],axis=0)[-1]))


(1.5)**(2./3)*0.98


jobs = []
TH2V, AV = np.meshgrid(np.linspace(0,2*np.pi,10), np.linspace(1.28,1.34,10))
for ii in range(len(TH2V.flatten())):
    jobs = jobs + [(ii,TH2V.flatten()[ii],AV.flatten()[ii])]


len(jobs)*.3/3600/16


import os, glob, pathlib
from loky import get_reusable_executor
import dill as pickle


executors_solve_ivp = get_reusable_executor(max_workers=16)
results = list(executors_solve_ivp.map(calc_FLIt, jobs))

np.savetxt("results.txt",results)



