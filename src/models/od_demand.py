from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ODDemand: #Store OD Demand matrix where
    #D[i,j] = tro[s generated from to origion i to destination j per ticks
    D: np.ndarray

    def __post_init__(self):
        if self.D.ndim !=2 or self.D.shape[0] != self.D.shape[1]:
            raise ValueError("OD Matrix must be square")
        if np.any(self.D < 0):
            raise ValueError("OD Matrix must be non negative")
    
    @property
    def n(self) -> int:
        return self.D.shape[0]
    def zero_diagonal(self) -> None:
        np.fill_diagonal(self.D, 0.0)
    def total_trips(self) -> float:
        return float(self.D.sum())
    def as_int(self) -> np.ndarray:
        return np.rint(self.D).astype(int)

def gravity_od(
    masses: np.ndarray,
    distances: np.ndarray,
    alpha: float = 1.0,
    beta: float = 2.0,
    scale: float = 1.0,
    eps: float = 1e-9,
    zero_diag: bool = True,) -> ODDemand:
    # SIMPLE GRAVITY MODEL
    masses = np.asarray(masses, dtype = float)
    distances = np.asarray(distances, dtype = float)

    if masses.ndim != 1:
        raise ValueError("Masses must be 1D")
    n = masses.size
    if distances.shape!= (n,n):
        raise ValueError("Distances must be (n,n)")
    
    mi = masses.reshape(n,1)
    mj = masses.reshape(1,n)
    numer = (mi ** alpha) * (mj ** alpha)
    denom = (distances ** beta) + eps

    D = scale * (numer/denom)

    od = ODDemand(D)
    if zero_diag:
        od.zero_diagonal()
    return od

def normalize_to_total_trips(od: ODDemand,
    total_trips: float) -> ODDemand: #scales OD so sum(D) = total_trips
    if total_trips < 0:
        raise ValueError("how can total_trips be negative?")
    current = od.total_trips()
    if current <= 0:
        return ODDemand(np.zeros_like(od.D))
    return ODDemand(od.D * (total_trips/ current))




