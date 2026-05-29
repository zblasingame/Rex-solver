"""
Butcher tableaus for deterministic Runge-Kutta ODE solvers.

Inspired by diffrax's solver design: https://github.com/patrick-kidger/diffrax

A Butcher tableau defines a Runge-Kutta method through:
    - c: nodes (time points for intermediate stages)
    - a: Runge-Kutta matrix (coefficients for intermediate stages)
    - b: weights (coefficients for final update)
    - b_error: error weights for adaptive stepping (optional)

The general form of an explicit RK method:
    k_i = f(t + c_i * h, y + h * sum_j(a_ij * k_j))
    y_{n+1} = y_n + h * sum_i(b_i * k_i)

References:
    - Butcher, J.C. (2008). Numerical Methods for Ordinary Differential Equations
    - Hairer, E., Nørsett, S.P., Wanner, G. (1993). Solving Ordinary Differential Equations I
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class ButcherTableau:
    """
    Butcher tableau representation for explicit Runge-Kutta methods.
    
    Attributes:
        c: Tensor of shape (s,) - nodes/time fractions
        a: Tensor of shape (s, s) - RK coefficient matrix (lower triangular for explicit)
        b: Tensor of shape (s,) - weights for final update
        b_error: Optional tensor of shape (s,) - error estimation weights
        order: Order of the method
        error_order: Order of the error estimate (for embedded methods)
        name: Human-readable name of the method
        is_fsal: Whether the method is First Same As Last (last k equals first k of next step)
    """
    c: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    b_error: Optional[torch.Tensor] = None
    order: int = 1
    error_order: Optional[int] = None
    name: str = "generic_rk"
    is_fsal: bool = False
    
    def __post_init__(self):
        """Validate tableau dimensions."""
        s = len(self.c)
        assert self.a.shape == (s, s), f"a must be ({s}, {s}), got {self.a.shape}"
        assert self.b.shape == (s,), f"b must be ({s},), got {self.b.shape}"
        if self.b_error is not None:
            assert self.b_error.shape == (s,), f"b_error must be ({s},), got {self.b_error.shape}"
    
    def to(self, device: torch.device, dtype: torch.dtype = None) -> "ButcherTableau":
        """Move tableau tensors to specified device and dtype."""
        dtype = dtype or self.c.dtype
        return ButcherTableau(
            c=self.c.to(device=device, dtype=dtype),
            a=self.a.to(device=device, dtype=dtype),
            b=self.b.to(device=device, dtype=dtype),
            b_error=self.b_error.to(device=device, dtype=dtype) if self.b_error is not None else None,
            order=self.order,
            error_order=self.error_order,
            name=self.name,
            is_fsal=self.is_fsal,
        )
    
    @property
    def num_stages(self) -> int:
        """Number of stages in the RK method."""
        return len(self.c)
    
    @property
    def is_explicit(self) -> bool:
        """Check if the method is explicit (lower triangular a matrix)."""
        return torch.allclose(self.a, torch.tril(self.a, diagonal=-1))
    
    @property
    def is_embedded(self) -> bool:
        """Check if the method has an embedded error estimate."""
        return self.b_error is not None


def Euler() -> ButcherTableau:
    """
    Forward Euler method (order 1).
    
    The simplest explicit RK method:
        y_{n+1} = y_n + h * f(t_n, y_n)
    """
    return ButcherTableau(
        c=torch.tensor([0.0]),
        a=torch.tensor([[0.0]]),
        b=torch.tensor([1.0]),
        order=1,
        name="euler",
    )


def Midpoint() -> ButcherTableau:
    """
    Explicit Midpoint method (order 2).
    
    Also known as modified Euler or RK2.
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h/2 * k1)
        y_{n+1} = y_n + h * k2
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 0.5]),
        a=torch.tensor([
            [0.0, 0.0],
            [0.5, 0.0],
        ]),
        b=torch.tensor([0.0, 1.0]),
        order=2,
        name="midpoint",
    )


def Heun() -> ButcherTableau:
    """
    Heun's method (order 2).
    
    Also known as improved Euler or explicit trapezoidal.
        k1 = f(t_n, y_n)
        k2 = f(t_n + h, y_n + h * k1)
        y_{n+1} = y_n + h/2 * (k1 + k2)
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 1.0]),
        a=torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
        ]),
        b=torch.tensor([0.5, 0.5]),
        order=2,
        name="heun",
    )


def Ralston() -> ButcherTableau:
    """
    Ralston's method (order 2).
    
    Optimal second-order method minimizing truncation error.
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 2.0/3.0]),
        a=torch.tensor([
            [0.0, 0.0],
            [2.0/3.0, 0.0],
        ]),
        b=torch.tensor([0.25, 0.75]),
        order=2,
        name="ralston",
    )


def SSPRK3() -> ButcherTableau:
    """
    Strong Stability Preserving RK3 (order 3).
    
    Optimal third-order SSP method, also known as Shu-Osher.
    Preserves total variation diminishing (TVD) property.
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 1.0, 0.5]),
        a=torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.25, 0.25, 0.0],
        ]),
        b=torch.tensor([1.0/6.0, 1.0/6.0, 2.0/3.0]),
        order=3,
        name="ssprk3",
    )


def RK4() -> ButcherTableau:
    """
    Classic 4th-order Runge-Kutta method.
    
    The most widely used RK method:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h/2 * k1)
        k3 = f(t_n + h/2, y_n + h/2 * k2)
        k4 = f(t_n + h, y_n + h * k3)
        y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 0.5, 0.5, 1.0]),
        a=torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]),
        b=torch.tensor([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]),
        order=4,
        name="rk4",
    )


def RK38() -> ButcherTableau:
    """
    3/8-rule RK4 method (order 4).
    
    Alternative fourth-order method with slightly different error characteristics.
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 1.0/3.0, 2.0/3.0, 1.0]),
        a=torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [1.0/3.0, 0.0, 0.0, 0.0],
            [-1.0/3.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 0.0],
        ]),
        b=torch.tensor([1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0]),
        order=4,
        name="rk38",
    )


def DOPRI5() -> ButcherTableau:
    """
    Dormand-Prince 5(4) method.
    
    Fifth-order method with embedded fourth-order error estimate.
    Used by scipy.integrate.odeint and MATLAB's ode45.
    Has the FSAL (First Same As Last) property.
    
    Reference: Dormand, J.R., Prince, P.J. (1980). A family of embedded
    Runge-Kutta formulae. J. Comp. Appl. Math. 6(1): 19-26.
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 0.2, 0.3, 0.8, 8.0/9.0, 1.0, 1.0]),
        a=torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
            [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
            [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
            [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
        ]),
        b=torch.tensor([35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]),
        b_error=torch.tensor([
            35.0/384.0 - 5179.0/57600.0,
            0.0,
            500.0/1113.0 - 7571.0/16695.0,
            125.0/192.0 - 393.0/640.0,
            -2187.0/6784.0 + 92097.0/339200.0,
            11.0/84.0 - 187.0/2100.0,
            -1.0/40.0,
        ]),
        order=5,
        error_order=4,
        name="dopri5",
        is_fsal=True,
    )


def Tsit5() -> ButcherTableau:
    """
    Tsitouras 5(4) method.
    
    Modern fifth-order method with good efficiency.
    
    Reference: Tsitouras, Ch. (2011). Runge-Kutta pairs of order 5(4)
    satisfying only the first column simplifying assumption.
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0]),
        a=torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.008480655492356989, 0.335480655492357, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.8971530571054935, -6.359448489975075, 4.3622954328695815, 0.0, 0.0, 0.0, 0.0],
            [5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525, 0.0, 0.0, 0.0],
            [5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383, 0.0, 0.0],
            [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0],
        ]),
        b=torch.tensor([0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0.0]),
        b_error=torch.tensor([
            0.09646076681806523 - 0.001780011052226,
            0.01 - 0.000816434459657,
            0.4798896504144996 - -0.007880878010262,
            1.379008574103742 - 0.144711007173263,
            -3.290069515436081 - -0.582357165452555,
            2.324710524099774 - 0.458082105929187,
            0.0 - 1.0/66.0,
        ]),
        order=5,
        error_order=4,
        name="tsit5",
        is_fsal=True,
    )


def Fehlberg45() -> ButcherTableau:
    """
    Fehlberg 4(5) method.
    
    Classical embedded RK method, uses 4th order for stepping with 5th order error estimate.
    Note: The "4(5)" means 4th order method with 5th order error estimate,
    which is inverted compared to DOPRI5's "5(4)".
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 0.25, 3.0/8.0, 12.0/13.0, 1.0, 0.5]),
        a=torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
            [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
            [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0, 0.0],
            [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0.0],
        ]),
        b=torch.tensor([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -0.2, 0.0]),
        b_error=torch.tensor([
            25.0/216.0 - 16.0/135.0,
            0.0,
            1408.0/2565.0 - 6656.0/12825.0,
            2197.0/4104.0 - 28561.0/56430.0,
            -0.2 + 9.0/50.0,
            0.0 - 2.0/55.0,
        ]),
        order=4,
        error_order=5,
        name="fehlberg45",
    )


def BogackiShampine() -> ButcherTableau:
    """
    Bogacki-Shampine 3(2) method.
    
    Efficient third-order method with embedded second-order error estimate.
    Has FSAL property. Used by MATLAB's ode23.
    
    Reference: Bogacki, P., Shampine, L.F. (1989). A 3(2) pair of
    Runge-Kutta formulas. Appl. Math. Lett. 2(4): 321-325.
    """
    return ButcherTableau(
        c=torch.tensor([0.0, 0.5, 0.75, 1.0]),
        a=torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.0, 0.0],
            [2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0],
        ]),
        b=torch.tensor([2.0/9.0, 1.0/3.0, 4.0/9.0, 0.0]),
        b_error=torch.tensor([
            2.0/9.0 - 7.0/24.0,
            1.0/3.0 - 0.25,
            4.0/9.0 - 1.0/3.0,
            0.0 - 0.125,
        ]),
        order=3,
        error_order=2,
        name="bogacki_shampine",
        is_fsal=True,
    )


# Registry of available RK methods
RK_TABLEAU_REGISTRY = {
    "euler": Euler,
    "midpoint": Midpoint,
    "heun": Heun,
    "ralston": Ralston,
    "rk4": RK4,
    "rk38": RK38,
    "ssprk3": SSPRK3,
    "dopri5": DOPRI5,
    "tsit5": Tsit5,
    "fehlberg45": Fehlberg45,
    "bogacki_shampine": BogackiShampine,
}


def get_rk_tableau(name: str) -> ButcherTableau:
    """
    Get a Butcher tableau by name.
    
    Args:
        name: Name of the RK method (case-insensitive)
        
    Returns:
        ButcherTableau for the specified method
        
    Raises:
        ValueError: If the method name is not recognized
    """
    name = name.lower()
    if name not in RK_TABLEAU_REGISTRY:
        available = ", ".join(RK_TABLEAU_REGISTRY.keys())
        raise ValueError(f"Unknown RK method '{name}'. Available: {available}")
    return RK_TABLEAU_REGISTRY[name]()


def list_rk_methods() -> list:
    """Return list of available RK method names."""
    return list(RK_TABLEAU_REGISTRY.keys())
