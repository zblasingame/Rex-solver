import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from typing import Callable, List, Literal, Optional, Union

from .rk_tableaus import ButcherTableau, get_rk_tableau, RK4, DOPRI5


# Some standard single-step ODE schemes Phi_h(t, x) of the form: x_n+1 = x_n + Phi_h(t_n, x_n)
def euler(model, t, x, h):
    return h * model(t, x)


def midpoint(model, t, x, h):
    return h * model(t + 0.5 * h, x + 0.5 * h * model(t, x))


def exp_midpoint(model, gamma, x, h):
    x_sig = x / _gamma_to_sigma(gamma)

    f1 = model(gamma, x)
    
    k2 = x_sig + h * 0.5 * f1
    f2 = model(gamma + 0.5 * h, _gamma_to_sigma(gamma + 0.5 * h) * k2)

    return h * f2


def exp_midpoint_chi(model, chi, x, h):
    x_alpha = x / _chi_to_alpha(chi)

    f1 = model(chi, x)
    
    k2 = x_alpha + h * 0.5 * f1
    f2 = model(chi + 0.5 * h, _chi_to_alpha(chi + 0.5 * h) * k2)

    return h * f2


def ralston3(model, t, x, h):
    k1 = model(t, x)
    k2 = model(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = model(t + 0.75 * h, x + 0.75 * h * k2)

    return h * (2/9 * k1 + 1/3 * k2 + 4/9 * k3)


def heun3(model, t, x, h):
    k1 = model(t, x)
    k2 = model(t + 1/3 * h, x + 1/3 * h * k1)
    k3 = model(t + 2/3 * h, x + 2/3 * h * k1)

    return h * (1/4 * k1 + 3/4 * k3)


def rk4(model, t, x, h):
    k1 = model(t, x)
    k2 = model(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = model(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = model(t + h, x + h * k3)

    return h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)


def exp_rk4(model, gamma, x, h):
    x_sig = x / _gamma_to_sigma(gamma)

    # sigmas cancel out
    f1 = model(gamma, x)

    k2 = x_sig + h * 0.5 * f1
    f2 = model(gamma + 0.5 * h, _gamma_to_sigma(gamma + 0.5 * h) * k2)

    k3 = x_sig + h * 0.5 * f2
    f3 = model(gamma + 0.5 * h, _gamma_to_sigma(gamma + 0.5 * h) * k2)

    k4 = x_sig + h * f3
    f4 = model(gamma + h, _gamma_to_sigma(gamma + h) * k3)

    return h * (1/6 * f1 + 1/3 * f2 + 1/3 * f3 + 1/6 * f4)


def exp_tsit5(model, gamma, x, h):
    x_sig = x / _gamma_to_sigma(gamma)

    # sigmas cancel out
    f1 = model(gamma, x)

    k2 = x_sig + 161 / 1000 * h * f1
    f2 = model(gamma + 161 / 1000 * h, _gamma_to_sigma(gamma + 161/1000 * h) * k2)

    k3 = x_sig - 0.8480655492356988544426874250230774675121177393430391537369234245294192976164141156943e-2 * h * f1\
            + 0.3354806554923569885444268742502307746751211773934303915373692342452941929761641411569 * h * f2
    f3 = model(gamma + 327 / 1000 * h, _gamma_to_sigma(gamma + 327 / 1000 * h) * k3)

    k4 = x_sig + 2.897153057105493432130432594192938764924887287701866490314866693455023795137503079289 * f1\
            - 6.359448489975074843148159912383825625952700647415626703305928850207288721235210244366 * f2\
            + 4.362295432869581411017727318190886861027813359713760212991062156752264926097707165077 * f3
    f4 = model(gamma + 9/10 * h, _gamma_to_sigma(gamma + 9/10 * h) * k3)

    k5 = x_sig + 5.325864828439256604428877920840511317836476253097040101202360397727981648835607691791 * f1\
            -11.74888356406282787774717033978577296188744178259862899288666928009020615663593781589 * f2\
            +7.495539342889836208304604784564358155658679161518186721010132816213648793440552049753 * f3\
            -0.9249506636175524925650207933207191611349983406029535244034750452930469056411389539635e-1 * f4
    f5 = model(gamma + 0.9800255409045096857298102862870245954942137979563024768854764293221195950761080302604 * h, _gamma_to_sigma(gamma + 0.9800255409045096857298102862870245954942137979563024768854764293221195950761080302604 * h) * k5)

    k6 = x_sig + 5.861455442946420028659251486982647890394337666164814434818157239052507339770711679748 * f1\
            - 12.92096931784710929170611868178335939541780751955743459166312250439928519268343184452 * f2\
            + 8.159367898576158643180400794539253485181918321135053305748355423955009222648673734986 * f3\
            - 0.7158497328140099722453054252582973869127213147363544882721139659546372402303777878835e-1 * f4\
            - 0.2826905039406838290900305721271224146717633626879770007617876201276764571291579142206e-1 * f5
    f6 = model(gamma + h, _gamma_to_sigma(gamma + h) * k6)

    k7 = x_sig + 0.9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1 * f1\
            + 1 / 100 * f2\
            + 0.479889650414499574775249532290596519913040462199033248833263494425454206015307452350 * f3\
            + 1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331 * f4\
            - 3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677 * f5\
            + 2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841 * f6
    f7 = model(gamma + h, _gamma_to_sigma(gamma + h) * k7)

    x1 = 0.9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1 * f1\
            + 1 / 100 * f2\
            + 0.4798896504144995747752495322905965199130404621990332488332634944254542060153074523509 * f3\
            + 1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331 * f4\
            - 3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677 * f5\
            + 2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841 * f6

    x2 = (0.9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1 - 0.9468075576583945807478876255758922856117527357724631226139574065785592789071067303271e-1) * f1\
            + (1/ 100 - 0.9183565540343253096776363936645313759813746240984095238905939532922955247253608687270e-2) * f2\
            + (0.4798896504144995747752495322905965199130404621990332488332634944254542060153074523509 - 0.4877705284247615707855642599631228241516691959761363774365216240304071651579571959813) * f3\
            + (1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331 - 1.234297566930478985655109673884237654035539930748192848315425833500484878378061439761) * f4\
            - (3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677 - 2.707712349983525454881109975059321670689605166938197378763992255714444407154902012702) * f5\
            + (2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841 - 1.866628418170587035753719399566211498666255505244122593996591602841258328965767580089) * f6\
            - 1 / 66 * f7

    return h * x1, h * x2


def ShARK(model, time_var, x, h, bm, pred_type='data'):
    t_to_w = _rho_to_siggamma if pred_type == 'data' else _chi_to_alpha

    x_sg = x / t_to_w(time_var)

    if pred_type == 'data':
        a, b = time_var, time_var + h
    else:
        a, b = time_var.pow(2), (time_var + h).pow(2)

    h_corr = b - a

    if h < 0:
        a, b = b, a

    # h_corr = h if pred_type == 'data' else (time_var + h).pow(2) - time_var.pow(2)

    W, U = bm(a, b, return_U=True)
    W, U = W.to(x.device), U.to(x.device)

    if h < 0:
        H = U / (-h_corr) - 0.5 * W
        W = -W
    else:
        H = U / h_corr - 0.5 * W

    Z1 = x_sg + H
    
    f1 = model(time_var, t_to_w(time_var) * Z1)

    Z2 = x_sg + h * (5/6) * f1 + (5/6) * W + H
    f2 = model(time_var + 5/6 * h, t_to_w(time_var + 5/6 * h) * Z2)

    return h * (0.4 * f1 + 0.6 * f2) + W



def euler_maruyama(model, time_var, x, h, bm, pred_type='data'):
    if pred_type == 'data':
        a, b = time_var, time_var + h
    else:
        a, b = time_var.pow(2), (time_var + h).pow(2)

    if h < 0:
        a, b = b, a

    W = bm(a, b).to(x.device)

    if h < 0:
        W = -W

    return h * model(time_var, x) + W


SOLVER_DICT = {
    'euler': euler,
    'midpoint': exp_midpoint,
    'rk4': exp_rk4,
    'tsit5': exp_tsit5,
    'euler_maruyama': euler_maruyama,
    'shark': ShARK,
}

SDE_SOLVERS = ['euler_maruyama', 'shark']


### Utility functions

def _t_to_sigma_alpha(scheduler, t, sched_type='linear'):
    """
    Assumes t in [eps, 1].
    """

    beta_0 = scheduler.betas[0] * 1000 # fix conversion between continuous time [0, 1] and discrete time {0,...,1000}
    beta_1 = scheduler.betas[-1] * 1000

    if sched_type == 'linear':
        delta = beta_1 - beta_0
        alpha_t = torch.exp(-delta/4 * t.pow(2) - beta_0/2 * t)
        sigma_t = torch.sqrt(1 - alpha_t.pow(2))

    elif sched_type == 'scaled_linear':
        alpha_t = torch.exp(-(beta_1 - 2 * torch.sqrt(beta_0 * beta_1) + beta_0) / 6 * t.pow(3)\
                - (torch.sqrt(beta_0 * beta_1) - beta_0) / 2 * t.pow(2)\
                - beta_0/2 * t)
        sigma_t = torch.sqrt(1 - alpha_t.pow(2))

    return alpha_t, sigma_t


def _gen_time_funcs(sched_type, rho=False, pred_type='data'):
    if not rho:
        if sched_type == 'linear':
            def _t_to_gamma(scheduler, t):
                """
                Assumes t in [eps, 1] and linear noise schedule.
                """

                beta_0 = scheduler.betas[0] * 1000
                beta_1 = scheduler.betas[-1] * 1000
                delta = beta_1 - beta_0

                alpha_t = torch.exp(-delta/4 * t.pow(2) - beta_0/2 * t)
                sigma_t = torch.sqrt(1 - alpha_t.pow(2))

                return alpha_t / sigma_t if pred_type == 'data' else sigma_t / alpha_t

            def _gamma_to_t(scheduler, gamma):
                """
                Assumes linear schedule!
                """

                beta_0 = scheduler.betas[0] * 1000
                beta_1 = scheduler.betas[-1] * 1000
                delta = beta_1 - beta_0

                p = -2 if pred_type == 'data' else 2

                return (-beta_0 + torch.sqrt(beta_0**2 + 2 * delta * torch.log(torch.pow(gamma, p) + 1.))) / delta

        elif sched_type == 'scaled_linear':
            def _t_to_gamma(scheduler, t):
                """
                Assumes t in [eps, 1] and scaled linear noise schedule.
                """

                beta_0 = scheduler.betas[0] * 1000
                beta_1 = scheduler.betas[-1] * 1000

                alpha_t = torch.exp(-(beta_1 - 2 * torch.sqrt(beta_0 * beta_1) + beta_0) / 6 * t.pow(3)\
                        - (torch.sqrt(beta_0 * beta_1) - beta_0) / 2 * t.pow(2)\
                        - beta_0/2 * t)
                sigma_t = torch.sqrt(1 - alpha_t.pow(2))

                return alpha_t / sigma_t if pred_type == 'data' else sigma_t / alpha_t

            def _gamma_to_t(scheduler, gamma):
                """
                Assumes scaled linear schedule!
                """

                beta_0 = scheduler.betas[0] * 1000
                beta_1 = scheduler.betas[-1] * 1000

                sq_b = torch.sqrt(beta_0 * beta_1)

                delta = beta_1 - 2 * sq_b + beta_0

                p = -2 if pred_type == 'data' else 2

                inner = 2 * (sq_b - beta_0).pow(3) - 3 * beta_0 * delta * (sq_b - beta_0) - 3 * delta.pow(2) * torch.log(torch.pow(gamma, p) + 1.)

                t = (beta_0 - sq_b + torch.pow(-inner, 1/3)) / delta

                return t

        return _t_to_gamma, _gamma_to_t

    else:
        if sched_type == 'linear':
            def _t_to_rho(scheduler, t):
                """
                Assumes t in [eps, 1] and linear noise schedule.
                """

                beta_0 = scheduler.betas[0] * 1000
                beta_1 = scheduler.betas[-1] * 1000
                delta = beta_1 - beta_0

                alpha_t = torch.exp(-delta/4 * t.pow(2) - beta_0/2 * t)
                sigma_t = torch.sqrt(1 - alpha_t.pow(2))

                return alpha_t.pow(2) / sigma_t.pow(2) if pred_type == 'data' else sigma_t / alpha_t

            def _rho_to_t(scheduler, rho):
                """
                Assumes linear schedule!
                """

                beta_0 = scheduler.betas[0] * 1000
                beta_1 = scheduler.betas[-1] * 1000
                delta = beta_1 - beta_0

                p = -1 if pred_type == 'data' else 2

                return (-beta_0 + torch.sqrt(beta_0**2 + 2 * delta * torch.log(torch.pow(rho, p) + 1.))) / delta

        elif sched_type == 'scaled_linear':
            def _t_to_rho(scheduler, t):
                """
                Assumes t in [eps, 1] and scaled linear noise schedule.
                """

                beta_0 = scheduler.betas[0] * 1000
                beta_1 = scheduler.betas[-1] * 1000
                delta = beta_1 - beta_0

                alpha_t = torch.exp(-(beta_1 - 2 * torch.sqrt(beta_0 * beta_1) + beta_0) / 6 * t.pow(3)\
                        - (torch.sqrt(beta_0 * beta_1) - beta_0) / 2 * t.pow(2)\
                        - beta_0/2 * t)
                sigma_t = torch.sqrt(1 - alpha_t.pow(2))

                return alpha_t.pow(2) / sigma_t.pow(2) if pred_type == 'data' else sigma_t / alpha_t

            def _rho_to_t(scheduler, rho):
                """
                Assumes scaled linear schedule!
                """

                beta_0 = scheduler.betas[0] * 1000
                beta_1 = scheduler.betas[-1] * 1000

                sq_b = torch.sqrt(beta_0 * beta_1)

                delta = beta_1 - 2 * sq_b + beta_0

                p = -1 if pred_type == 'data' else 2

                inner = 2 * (sq_b - beta_0).pow(3) - 3 * beta_0 * delta * (sq_b - beta_0) - 3 * delta.pow(2) * torch.log(torch.pow(rho, p) + 1.)

                t = (beta_0 - sq_b + torch.pow(-inner, 1/3)) / delta

                return t

        return _t_to_rho, _rho_to_t


def _gamma_to_sigma(gamma):
    """
    Assumes VP schedule
    """

    return torch.rsqrt(gamma.pow(2) + 1.)


def _chi_to_alpha(chi):
    """
    Assumes VP schedule
    """

    return torch.rsqrt(chi.pow(2) + 1.)


def _rho_to_sigma(rho):
    """
    Assumes VP schedule
    """

    return torch.rsqrt(rho + 1.)


def _rho_to_siggamma(rho):
    """
    Assumes VP schedule
    """

    return torch.sqrt((1. / rho) + 1.) / (rho + 1.)


def _lambda_to_sigma(lamb):
    """
    Assumes VP schedule
    """

    return torch.rsqrt(torch.exp(2. * lamb) + 1.)
    

def _t_to_lambda(scheduler, t):
    """
    Assumes t in [eps, 1] and linear noise schedule.
    """

    beta_0 = scheduler.betas[0] * 1000
    beta_1 = scheduler.betas[-1] * 1000
    delta = beta_1 - beta_0

    alpha_t = torch.exp(-delta/4 * t.pow(2) - beta_0/2 * t)
    sigma_t = torch.sqrt(1 - alpha_t.pow(2))

    return torch.log(alpha_t) - torch.log(sigma_t)


def _lambda_to_t(scheduler, lamb):
    """
    Code borrowed from https://github.com/LuChengTHU/dpm-solver/blob/main/dpm_solver_pytorch.py
    """
    beta_0 = scheduler.betas[0] * 1000
    beta_1 = scheduler.betas[-1] * 1000

    tmp = 2. * (beta_1 - beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
    Delta = beta_0**2 + tmp
    return tmp / (torch.sqrt(Delta) + beta_0) / (beta_1 - beta_0)


def _convert_noise_to_data(scheduler, model, t, x, sched_type='linear'):
    alpha_t, sigma_t = _t_to_sigma_alpha(scheduler, t, sched_type=sched_type)

    return (x - sigma_t * model(t, x)) / alpha_t


def psi(model_func, scheduler, xt, timesteps, solver='euler', low_order_final_n_steps=0, bm=None, pred_type='data', sched_type='linear'):
    """
    Lawson applied to Phi
    """

    # Choose underlying solver
    is_sde = (solver in SDE_SOLVERS)
    psi = SOLVER_DICT[solver]

    if not is_sde:
        _t_to_gamma, _gamma_to_t = _gen_time_funcs(sched_type=sched_type, pred_type=pred_type)
        t_to_gamma = _t_to_gamma
        gamma_to_t = _gamma_to_t
        gamma_to_sigma = _gamma_to_sigma if pred_type == 'data' else _chi_to_alpha
    else:
        _t_to_rho, _rho_to_t = _gen_time_funcs(sched_type=sched_type, rho=True, pred_type=pred_type)
        t_to_gamma = _t_to_rho
        gamma_to_t = _rho_to_t
        gamma_to_sigma = _rho_to_siggamma if pred_type == 'data' else _chi_to_alpha


    # create timesteps in gamma, alt gamma^2 = rho for SDEs
    gammas = t_to_gamma(scheduler, timesteps)

    # Push gamma reparam back to time t and convert noise pred to data pred
    if pred_type == 'data':
        wrap_model = lambda gamma, x: _convert_noise_to_data(scheduler, model_func, gamma_to_t(scheduler, gamma), x, sched_type=sched_type)
    else:
        p = 2 if is_sde else 1
        wrap_model = lambda gamma, x: p * model_func(gamma_to_t(scheduler, gamma), x)

    for n in tqdm(range(len(gammas)-1)):
        gamma_n = gammas[n]
        gamma_n1 = gammas[n+1]
        h = gamma_n1 - gamma_n

        sigma_n = gamma_to_sigma(gamma_n)
        sigma_n1 = gamma_to_sigma(gamma_n1)

        if n < (len(gammas) - 1 - low_order_final_n_steps):
            if not is_sde:
                _psi = lambda t, x, h: psi(wrap_model, t, x, h)
            else:
                _psi = lambda t, x, h: psi(wrap_model, t, x, h, bm, pred_type=pred_type)
        else:
            _psi = lambda t, x, h: euler(wrap_model, t, x, h)

        xt = (sigma_n1 / sigma_n) * xt + sigma_n1 * _psi(gamma_n, xt, h)

    return xt



def rex_forward(model_func, scheduler, xt, xt_hat, timesteps, solver='euler', coupling=0.999, low_order_final_n_steps=0, bm=None, pred_type='data', sched_type='linear'):
    """
    Fixed-step Rex forward sweep on a user-supplied diffusers ``timesteps`` grid.

    Based on McCallum & Foster's reversible ODE solver and adapted for diffusion
    models. Pairs with :func:`rex_backward` to obtain an algebraic inverse.

    .. deprecated::
        Prefer :class:`RexTorchdynWrapper` for new code: it exposes the same
        algorithm, supports embedded RK tableaux from
        :mod:`samplers.rk_tableaus`, and offers an adaptive step-size mode.
        ``rex_forward`` / ``rex_backward`` are kept because the
        ``sd_sampling.py``, ``interpolate.py``, ``image_editing.py`` and
        ``celeba.py`` scripts still use them and their numerics match the
        published results.
    """

    # Choose underlying solver
    is_sde = (solver in SDE_SOLVERS)
    psi = SOLVER_DICT[solver]

    if not is_sde:
        _t_to_gamma, _gamma_to_t = _gen_time_funcs(sched_type=sched_type, pred_type=pred_type)
        t_to_gamma = _t_to_gamma
        gamma_to_t = _gamma_to_t
        gamma_to_sigma = _gamma_to_sigma if pred_type == 'data' else _chi_to_alpha
    else:
        _t_to_rho, _rho_to_t = _gen_time_funcs(sched_type=sched_type, rho=True, pred_type=pred_type)
        t_to_gamma = _t_to_rho
        gamma_to_t = _rho_to_t
        gamma_to_sigma = _rho_to_siggamma if pred_type == 'data' else _chi_to_alpha


    # create timesteps in gamma, alt gamma^2 = rho for SDEs
    gammas = t_to_gamma(scheduler, timesteps)

    # Push gamma reparam back to time t and convert noise pred to data pred
    if pred_type == 'data':
        wrap_model = lambda gamma, x: _convert_noise_to_data(scheduler, model_func, gamma_to_t(scheduler, gamma), x, sched_type=sched_type)
    else:
        p = 2 if is_sde else 1
        wrap_model = lambda gamma, x: p * model_func(gamma_to_t(scheduler, gamma), x)

    # xt.to(torch.float32)
    # xt_hat.to(torch.float32)

    for n in tqdm(range(len(gammas)-1)):
        gamma_n = gammas[n]
        gamma_n1 = gammas[n+1]
        h = gamma_n1 - gamma_n

        sigma_n = gamma_to_sigma(gamma_n)
        sigma_n1 = gamma_to_sigma(gamma_n1)

        if n < (len(gammas) - 1 - low_order_final_n_steps):
            if not is_sde:
                _psi = lambda t, x, h: psi(wrap_model, t, x, h)
            else:
                _psi = lambda t, x, h: psi(wrap_model, t, x, h, bm, pred_type=pred_type)
        else:
            if not is_sde:
                _psi = lambda t, x, h: euler(wrap_model, t, x, h)
            else:
                _psi = lambda t, x, h: euler_maruyama(wrap_model, t, x, h, bm, pred_type=pred_type)


        xt = (sigma_n1 / sigma_n) * (coupling * xt + (1-coupling) * xt_hat) + sigma_n1 * _psi(gamma_n, xt_hat, h)
        xt_hat = (sigma_n1 / sigma_n) * xt_hat - sigma_n1 * _psi(gamma_n1, xt, -h)

    return xt, xt_hat


def rex_backward(model_func, scheduler, xt, xt_hat, timesteps, solver='euler', coupling=0.999, low_order_final_n_steps=0, bm=None, pred_type='data', sched_type='linear'):
    """
    Fixed-step Rex backward sweep on a user-supplied diffusers ``timesteps`` grid.

    Inverts a forward sweep produced by :func:`rex_forward` on the same grid.

    .. deprecated::
        Prefer :class:`RexTorchdynWrapper` for new code (see the note on
        :func:`rex_forward`).
    """

    # Choose underlying solver
    is_sde = (solver in SDE_SOLVERS)
    psi = SOLVER_DICT[solver]

    if not is_sde:
        _t_to_gamma, _gamma_to_t = _gen_time_funcs(sched_type=sched_type, pred_type=pred_type)
        t_to_gamma = _t_to_gamma
        gamma_to_t = _gamma_to_t
        gamma_to_sigma = _gamma_to_sigma if pred_type == 'data' else _chi_to_alpha
    else:
        _t_to_rho, _rho_to_t = _gen_time_funcs(sched_type=sched_type, rho=True, pred_type=pred_type)
        t_to_gamma = _t_to_rho
        gamma_to_t = _rho_to_t
        gamma_to_sigma = _rho_to_siggamma if pred_type == 'data' else _chi_to_alpha


    # create timesteps in gamma, alt gamma^2 = rho for SDEs
    gammas = t_to_gamma(scheduler, timesteps)

    # Push gamma reparam back to time t and convert noise pred to data pred
    if pred_type == 'data':
        wrap_model = lambda gamma, x: _convert_noise_to_data(scheduler, model_func, gamma_to_t(scheduler, gamma), x, sched_type=sched_type)
    else:
        p = 2 if is_sde else 1
        wrap_model = lambda gamma, x: p * model_func(gamma_to_t(scheduler, gamma), x)

    # xt.to(torch.float32)
    # xt_hat.to(torch.float32)

    coupling_inv = 1. / coupling

    for n in tqdm(range(len(gammas) - 2, -1, -1)):
        gamma_n = gammas[n]
        gamma_n1 = gammas[n+1]
        h = gamma_n1 - gamma_n

        sigma_n = gamma_to_sigma(gamma_n)
        sigma_n1 = gamma_to_sigma(gamma_n1)

        if n < (len(gammas) - 1 - low_order_final_n_steps):
            if not is_sde:
                _psi = lambda t, x, h: psi(wrap_model, t, x, h)
            else:
                _psi = lambda t, x, h: psi(wrap_model, t, x, h, bm, pred_type=pred_type)
        else:
            if not is_sde:
                _psi = lambda t, x, h: euler(wrap_model, t, x, h)
            else:
                _psi = lambda t, x, h: euler_maruyama(wrap_model, t, x, h, bm, pred_type=pred_type)

        xt_hat = (sigma_n / sigma_n1) * xt_hat + sigma_n * _psi(gamma_n1, xt, -h)
        xt = (sigma_n / sigma_n1) * (coupling_inv * xt) + (1 - coupling_inv) * xt_hat - sigma_n * coupling_inv * _psi(gamma_n, xt_hat, h)

    return xt, xt_hat


# ---------------------------------------------------------------------------
# RexTorchdynWrapper — canonical Rex solver (fixed-step + adaptive)
# ---------------------------------------------------------------------------
#
# This class is the reference implementation used in the paper. It supports
# both fixed-step and embedded-error adaptive stepping, both VP and
# flow-matching schedules, and any Butcher tableau registered in
# samplers.rk_tableaus.
#
# Lifted verbatim (modulo cosmetics) from scripts/inversion.py so that
# scripts/image_editing_rex.py and scripts/inversion.py can share it.


class RexTorchdynWrapper(nn.Module):
    """
    Rex (Reversible Exponential) solver for diffusion models.

    Combines:
      1. Exponential RK methods         (handle the linear drift)
      2. McCallum-Foster reversible coupling  (algebraic reversibility)
    """

    def __init__(
        self,
        model: nn.Module,
        tableau: Union[ButcherTableau, str, None] = None,
        n_steps: int = 100,
        schedule: str = "flow_matching",
        alpha_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sigma_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        prediction_type: str = "data",
        model_type: str = "velocity",
        zeta: float = 0.5,
        adaptive: bool = False,
        step_domain: Literal["t", "varsigma"] = "t",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        eps: float = 1e-5,
        min_step: float = 1e-10,
        max_step: Optional[float] = None,
        safety_factor: float = 0.9,
        min_factor: float = 0.1,
        max_factor: float = 2.0,
        scheduler=None,
        sched_type: str = "scaled_linear",
    ):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.schedule = schedule
        self.prediction_type = prediction_type
        self.model_type = model_type
        self.zeta = zeta
        self.adaptive = adaptive
        self.step_domain = step_domain
        self.atol = atol
        self.rtol = rtol
        self.eps = eps
        self.min_step = min_step
        self.max_step = max_step
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.nfe = 0
        self.scheduler = scheduler
        self.sched_type = sched_type

        if tableau is None:
            self.tableau = RK4() if not adaptive else DOPRI5()
        elif isinstance(tableau, str):
            self.tableau = get_rk_tableau(tableau)
        else:
            self.tableau = tableau

        if adaptive and not self.tableau.is_embedded:
            raise ValueError(
                f"Adaptive stepping requires an embedded RK method. "
                f"'{self.tableau.name}' does not have embedded error coefficients."
            )

        self._setup_schedule(schedule, alpha_fn, sigma_fn)

    # ----------------------------------------------------------
    def _setup_schedule(self, schedule, alpha_fn, sigma_fn):
        if schedule == "flow_matching":
            self.alpha_fn = lambda t: t
            self.sigma_fn = lambda t: 1.0 - t
            self._has_closed_form_inverse = True
            self._vp_schedule = False
        elif schedule in ("vp", "custom"):
            if schedule == "vp" and self.scheduler is None:
                raise ValueError("VP schedule requires a scheduler")
            if schedule == "custom" and alpha_fn is None:
                raise ValueError("Custom schedule requires alpha_fn and sigma_fn")
            if alpha_fn is not None and sigma_fn is not None:
                self.alpha_fn = alpha_fn
                self.sigma_fn = sigma_fn
            else:
                self.alpha_fn = self._get_alpha_from_scheduler
                self.sigma_fn = self._get_sigma_from_scheduler
            self._has_closed_form_inverse = True
            self._vp_schedule = True
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def _get_alpha_from_scheduler(self, t):
        return self._t_to_alpha_sigma(t)[0]

    def _get_sigma_from_scheduler(self, t):
        return self._t_to_alpha_sigma(t)[1]

    def _t_to_alpha_sigma(self, t):
        beta_0 = self.scheduler.betas[0] * 1000
        beta_1 = self.scheduler.betas[-1] * 1000
        if self.sched_type == "linear":
            delta = beta_1 - beta_0
            alpha_t = torch.exp(-delta / 4 * t.pow(2) - beta_0 / 2 * t)
        elif self.sched_type == "scaled_linear":
            alpha_t = torch.exp(
                -(beta_1 - 2 * torch.sqrt(beta_0 * beta_1) + beta_0) / 6 * t.pow(3)
                - (torch.sqrt(beta_0 * beta_1) - beta_0) / 2 * t.pow(2)
                - beta_0 / 2 * t
            )
        else:
            raise ValueError(f"Unknown sched_type: {self.sched_type}")
        sigma_t = torch.sqrt(1 - alpha_t.pow(2))
        return alpha_t, sigma_t

    # ----------------------------------------------------------
    def _convert_model_output(self, t, x, v):
        alpha = self.alpha_fn(t)
        sigma = self.sigma_fn(t)
        if self.prediction_type == "data":
            if self.model_type == "velocity":
                return x + sigma * v
            elif self.model_type == "noise":
                return (x - sigma * v) / torch.clamp(alpha, min=self.eps)
            elif self.model_type == "data":
                return v
        elif self.prediction_type == "noise":
            if self.model_type == "velocity":
                return x - alpha * v
            elif self.model_type == "data":
                return (x - alpha * v) / torch.clamp(sigma, min=self.eps)
            elif self.model_type == "noise":
                return v
        raise ValueError(
            f"Invalid combination: prediction_type={self.prediction_type}, "
            f"model_type={self.model_type}"
        )

    def _get_weight(self, t):
        if self.prediction_type == "data":
            return self.sigma_fn(t)
        return self.alpha_fn(t)

    def _get_time_variable(self, t):
        alpha = self.alpha_fn(t)
        sigma = self.sigma_fn(t)
        if self.prediction_type == "data":
            return alpha / torch.clamp(sigma, min=self.eps)
        return sigma / torch.clamp(alpha, min=self.eps)

    def _inverse_time_variable(self, gamma):
        if not self._vp_schedule:
            if self.prediction_type == "data":
                t = gamma / (1.0 + gamma)
            else:
                t = 1.0 / (1.0 + gamma)
            return t.clamp(0.0, 1.0)
        return self._gamma_to_t_vp(gamma)

    def _gamma_to_t_vp(self, gamma):
        beta_0 = self.scheduler.betas[0] * 1000
        beta_1 = self.scheduler.betas[-1] * 1000
        p = -2 if self.prediction_type == "data" else 2
        if self.sched_type == "linear":
            delta = beta_1 - beta_0
            inner = beta_0 ** 2 + 2 * delta * torch.log(torch.pow(gamma, p) + 1.0)
            t = (-beta_0 + torch.sqrt(inner)) / delta
        elif self.sched_type == "scaled_linear":
            sq_b = torch.sqrt(beta_0 * beta_1)
            delta = beta_1 - 2 * sq_b + beta_0
            inner = (
                2 * (sq_b - beta_0).pow(3)
                - 3 * beta_0 * delta * (sq_b - beta_0)
                - 3 * delta.pow(2) * torch.log(torch.pow(gamma, p) + 1.0)
            )
            t = (beta_0 - sq_b + torch.pow(-inner, 1 / 3)) / delta
        else:
            raise ValueError(f"Unknown sched_type: {self.sched_type}")
        return t.clamp(self.eps, 1.0 - self.eps)

    # ----------------------------------------------------------
    def _psi_step(self, t_start, t_end, x):
        device, dtype = x.device, x.dtype
        tableau = self.tableau.to(device, dtype)
        s = tableau.num_stages

        if not isinstance(t_start, torch.Tensor):
            t_start = torch.tensor([t_start], device=device, dtype=dtype)
        if not isinstance(t_end, torch.Tensor):
            t_end = torch.tensor([t_end], device=device, dtype=dtype)

        w_start = self._get_weight(t_start)
        w_end = self._get_weight(t_end)
        zeta_start = self._get_time_variable(t_start)
        zeta_end = self._get_time_variable(t_end)
        h = zeta_end - zeta_start

        k: List[torch.Tensor] = []
        for i in range(s):
            zeta_i = zeta_start + tableau.c[i] * h
            t_i = self._inverse_time_variable(zeta_i)
            w_i = self._get_weight(t_i)

            Z_i = x / torch.clamp(w_start, min=self.eps)
            for j in range(i):
                if tableau.a[i, j] != 0:
                    Z_i = Z_i + h * tableau.a[i, j] * k[j]

            scaled_state = w_i * Z_i
            self.nfe += 1
            model_out = self.model(t_i, scaled_state)
            k_i = self._convert_model_output(t_i, scaled_state, model_out)
            k.append(k_i)

        increment = torch.zeros_like(x)
        for i in range(s):
            if tableau.b[i] != 0:
                increment = increment + h * tableau.b[i] * k[i]

        error = None
        if tableau.b_error is not None:
            err_incr = torch.zeros_like(x)
            for i in range(s):
                if tableau.b_error[i] != 0:
                    err_incr = err_incr + h * tableau.b_error[i] * k[i]
            error = torch.abs(w_end * err_incr).max()

        return increment, error

    # ----------------------------------------------------------
    def _compute_step_factor(self, error, x):
        scale = self.atol + self.rtol * torch.abs(x).max()
        error_ratio = error.item() / scale.item()
        accept = error_ratio <= 1.0
        if error_ratio == 0:
            factor = self.max_factor
        elif accept:
            factor = self.safety_factor * (1.0 / error_ratio) ** (1.0 / (self.tableau.order + 1))
        else:
            factor = self.safety_factor * (1.0 / error_ratio) ** (1.0 / self.tableau.order)
        factor = max(self.min_factor, min(self.max_factor, factor))
        return factor, accept

    # ----------------------------------------------------------
    def _forward_step(self, t_n, t_n1, x_n, x_hat_n):
        device, dtype = x_n.device, x_n.dtype
        if not isinstance(t_n, torch.Tensor):
            t_n = torch.tensor([t_n], device=device, dtype=dtype)
        if not isinstance(t_n1, torch.Tensor):
            t_n1 = torch.tensor([t_n1], device=device, dtype=dtype)

        w_n = self._get_weight(t_n)
        w_n1 = self._get_weight(t_n1)
        weight_ratio = w_n1 / torch.clamp(w_n, min=self.eps)

        psi_h, error = self._psi_step(t_n, t_n1, x_hat_n)
        x_n1 = weight_ratio * (self.zeta * x_n + (1.0 - self.zeta) * x_hat_n) + w_n1 * psi_h

        psi_neg_h, _ = self._psi_step(t_n1, t_n, x_n1)
        x_hat_n1 = weight_ratio * x_hat_n - w_n1 * psi_neg_h

        return x_n1, x_hat_n1, error

    def _backward_step(self, t_n, t_n1, x_n1, x_hat_n1):
        device, dtype = x_n1.device, x_n1.dtype
        if not isinstance(t_n, torch.Tensor):
            t_n = torch.tensor([t_n], device=device, dtype=dtype)
        if not isinstance(t_n1, torch.Tensor):
            t_n1 = torch.tensor([t_n1], device=device, dtype=dtype)

        w_n = self._get_weight(t_n)
        w_n1 = self._get_weight(t_n1)
        weight_ratio_inv = w_n / torch.clamp(w_n1, min=self.eps)
        zeta_inv = 1.0 / self.zeta

        psi_neg_h, _ = self._psi_step(t_n1, t_n, x_n1)
        x_hat_n = weight_ratio_inv * x_hat_n1 + w_n * psi_neg_h

        psi_h, error = self._psi_step(t_n, t_n1, x_hat_n)
        x_n = (
            weight_ratio_inv * zeta_inv * x_n1
            + (1.0 - zeta_inv) * x_hat_n
            - w_n * zeta_inv * psi_h
        )
        return x_n, x_hat_n, error

    # ----------------------------------------------------------
    def forward_solve(self, x, x_hat, t_span):
        t_start, t_end = t_span[0].item(), t_span[-1].item()
        if self.adaptive:
            if self.step_domain == "t":
                return self._forward_solve_t_domain(x, x_hat, t_start, t_end)
            return self._forward_solve_varsigma_domain(x, x_hat, t_start, t_end)
        # Fixed stepping
        h_t = (t_end - t_start) / self.n_steps
        t_current = t_start
        for _ in range(self.n_steps):
            t_next = t_current + h_t
            x, x_hat, _ = self._forward_step(t_current, t_next, x, x_hat)
            t_current = t_next
        return x, x_hat

    def backward_solve(self, x, x_hat, t_span):
        t_start, t_end = t_span[0].item(), t_span[-1].item()
        if self.adaptive:
            if self.step_domain == "t":
                return self._backward_solve_t_domain(x, x_hat, t_start, t_end)
            return self._backward_solve_varsigma_domain(x, x_hat, t_start, t_end)
        # Fixed stepping
        h_t = (t_end - t_start) / self.n_steps
        t_current = t_start
        for _ in range(self.n_steps):
            t_prev = t_current + h_t
            x, x_hat, _ = self._backward_step(t_prev, t_current, x, x_hat)
            t_current = t_prev
        return x, x_hat

    # --- adaptive helpers ---
    def _forward_solve_t_domain(self, x, x_hat, t_start, t_end):
        direction = 1.0 if t_end > t_start else -1.0
        t_current = t_start
        h_t = abs(t_end - t_start) / self.n_steps
        while direction * (t_end - t_current) > self.eps:
            t_next = t_current + direction * h_t
            if direction * (t_next - t_end) > 0:
                t_next = t_end
            x_new, x_hat_new, error = self._forward_step(t_current, t_next, x, x_hat)
            if error is not None:
                factor, accept = self._compute_step_factor(error, x)
                if accept:
                    x, x_hat = x_new, x_hat_new
                    t_current = t_next
                h_t = max(self.min_step, min(self.max_step or h_t * 10, h_t * factor))
            else:
                x, x_hat = x_new, x_hat_new
                t_current = t_next
        return x, x_hat

    def _forward_solve_varsigma_domain(self, x, x_hat, t_start, t_end):
        device, dtype = x.device, x.dtype
        vs_s = self._get_time_variable(torch.tensor([t_start], device=device, dtype=dtype)).item()
        vs_e = self._get_time_variable(torch.tensor([t_end], device=device, dtype=dtype)).item()
        direction = 1.0 if vs_e > vs_s else -1.0
        vs_cur = vs_s
        h_vs = abs(vs_e - vs_s) / self.n_steps
        while direction * (vs_e - vs_cur) > self.eps:
            vs_nxt = vs_cur + direction * h_vs
            if direction * (vs_nxt - vs_e) > 0:
                vs_nxt = vs_e
            t_c = self._inverse_time_variable(torch.tensor([vs_cur], device=device, dtype=dtype))
            t_n = self._inverse_time_variable(torch.tensor([vs_nxt], device=device, dtype=dtype))
            x_new, x_hat_new, error = self._forward_step(t_c, t_n, x, x_hat)
            if error is not None:
                factor, accept = self._compute_step_factor(error, x)
                if accept:
                    x, x_hat = x_new, x_hat_new
                    vs_cur = vs_nxt
                h_vs = max(self.min_step, min(self.max_step or h_vs * 10, h_vs * factor))
            else:
                x, x_hat = x_new, x_hat_new
                vs_cur = vs_nxt
        return x, x_hat

    def _backward_solve_t_domain(self, x, x_hat, t_start, t_end):
        direction = 1.0 if t_end > t_start else -1.0
        t_current = t_start
        h_t = abs(t_end - t_start) / self.n_steps
        while direction * (t_end - t_current) > self.eps:
            t_prev = t_current + direction * h_t
            if direction * (t_prev - t_end) > 0:
                t_prev = t_end
            x_new, x_hat_new, error = self._backward_step(t_prev, t_current, x, x_hat)
            if error is not None:
                factor, accept = self._compute_step_factor(error, x)
                if accept:
                    x, x_hat = x_new, x_hat_new
                    t_current = t_prev
                h_t = max(self.min_step, min(self.max_step or h_t * 10, h_t * factor))
            else:
                x, x_hat = x_new, x_hat_new
                t_current = t_prev
        return x, x_hat

    def _backward_solve_varsigma_domain(self, x, x_hat, t_start, t_end):
        device, dtype = x.device, x.dtype
        vs_s = self._get_time_variable(torch.tensor([t_start], device=device, dtype=dtype)).item()
        vs_e = self._get_time_variable(torch.tensor([t_end], device=device, dtype=dtype)).item()
        direction = 1.0 if vs_e > vs_s else -1.0
        vs_cur = vs_s
        h_vs = abs(vs_e - vs_s) / self.n_steps
        while direction * (vs_e - vs_cur) > self.eps:
            vs_prv = vs_cur + direction * h_vs
            if direction * (vs_prv - vs_e) > 0:
                vs_prv = vs_e
            t_c = self._inverse_time_variable(torch.tensor([vs_cur], device=device, dtype=dtype))
            t_p = self._inverse_time_variable(torch.tensor([vs_prv], device=device, dtype=dtype))
            x_new, x_hat_new, error = self._backward_step(t_p, t_c, x, x_hat)
            if error is not None:
                factor, accept = self._compute_step_factor(error, x)
                if accept:
                    x, x_hat = x_new, x_hat_new
                    vs_cur = vs_prv
                h_vs = max(self.min_step, min(self.max_step or h_vs * 10, h_vs * factor))
            else:
                x, x_hat = x_new, x_hat_new
                vs_cur = vs_prv
        return x, x_hat


# ---------------------------------------------------------------------------
# Factory: build a RexTorchdynWrapper around a Stable Diffusion UNet
# ---------------------------------------------------------------------------

def create_rex_solver(
    model,
    tableau="rk4",
    n_steps=50,
    prediction_type="data",
    zeta=0.5,
    adaptive=False,
    step_domain="t",
    atol=1e-5,
    rtol=1e-5,
    scheduler=None,
    sched_type="scaled_linear",
) -> "RexTorchdynWrapper":
    """
    Build a :class:`RexTorchdynWrapper` configured for a VP (DDPM/SD) schedule.

    ``model`` is expected to expose the signature ``model(t, x) -> noise``
    (i.e. ``model_type='noise'``). The schedule's ``alpha`` / ``sigma``
    closed forms are derived from the supplied ``scheduler.betas`` (a
    diffusers scheduler with ``beta_schedule == sched_type``).
    """
    if scheduler is None:
        raise ValueError("create_rex_solver requires a diffusers scheduler.")

    beta_0 = scheduler.betas[0].item() * 1000
    beta_1 = scheduler.betas[-1].item() * 1000

    if sched_type == "scaled_linear":
        def alpha_fn(t):
            if isinstance(t, (int, float)):
                t = torch.tensor([t])
            sq_b = torch.sqrt(torch.tensor(beta_0 * beta_1))
            return torch.exp(
                -(beta_1 - 2 * sq_b + beta_0) / 6 * t.pow(3)
                - (sq_b - beta_0) / 2 * t.pow(2)
                - beta_0 / 2 * t
            )
    else:
        delta = beta_1 - beta_0
        def alpha_fn(t):
            if isinstance(t, (int, float)):
                t = torch.tensor([t])
            return torch.exp(-delta / 4 * t.pow(2) - beta_0 / 2 * t)

    def sigma_fn(t):
        return torch.sqrt(1 - alpha_fn(t).pow(2))

    return RexTorchdynWrapper(
        model=model,
        tableau=tableau,
        n_steps=n_steps,
        schedule="vp",
        alpha_fn=alpha_fn,
        sigma_fn=sigma_fn,
        prediction_type=prediction_type,
        model_type="noise",
        zeta=zeta,
        adaptive=adaptive,
        step_domain=step_domain,
        atol=atol,
        rtol=rtol,
        eps=1e-5,
        scheduler=scheduler,
        sched_type=sched_type,
    )
