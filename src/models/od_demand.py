"""
Origin-Destination (OD) matrix modeling and generation.

Implements gravity models, doubly-constrained balancing, and OD demand
manipulation for transportation planning and traffic simulation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ODDemand:
    """
    Origin-Destination demand matrix for trip generation.
    
    Stores a square matrix where D[i,j] represents trips generated from
    origin i to destination j per simulation tick.
    
    Attributes:
        D: n×n numpy array of trip demands (non-negative)
    
    Raises:
        ValueError: If matrix is not square or contains negative values
    
    Example:
        >>> demand = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]])
        >>> od = ODDemand(demand)
        >>> od.total_trips()
        46.0
        >>> od.zero_diagonal()
        >>> od.total_trips()
        36.0
    """
    
    D: np.ndarray

    def __post_init__(self) -> None:
        """Validate OD matrix properties."""
        if self.D.ndim != 2:
            raise ValueError(
                f"OD matrix must be 2-dimensional, got {self.D.ndim}D"
            )
        
        if self.D.shape[0] != self.D.shape[1]:
            raise ValueError(
                f"OD matrix must be square, got shape {self.D.shape}"
            )
        
        if np.any(self.D < 0):
            raise ValueError(
                "OD matrix cannot contain negative values"
            )
        
        # Ensure float type for numerical stability
        self.D = self.D.astype(float)
    
    @property
    def n(self) -> int:
        """Number of zones (matrix dimension)."""
        return self.D.shape[0]
    
    def zero_diagonal(self) -> None:
        """Set diagonal elements to zero (no intra-zonal trips)."""
        np.fill_diagonal(self.D, 0.0)
    
    def total_trips(self) -> float:
        """Calculate total number of trips across all OD pairs."""
        return float(self.D.sum())
    
    def as_int(self) -> np.ndarray:
        """
        Convert trip values to integers via rounding.
        
        Returns:
            Integer array with rounded trip values
        """
        return np.rint(self.D).astype(int)
    
    def production(self) -> np.ndarray:
        """
        Trip production (outflows) from each zone.
        
        Returns:
            1D array where element i is sum of row i (trips originating from i)
        
        Example:
            >>> prod = od.production()
            >>> print(f"Zone 0 produces {prod[0]:.0f} trips")
        """
        return np.sum(self.D, axis=1)
    
    def attraction(self) -> np.ndarray:
        """
        Trip attraction (inflows) to each zone.
        
        Returns:
            1D array where element j is sum of column j (trips destined for j)
        
        Example:
            >>> attr = od.attraction()
            >>> print(f"Zone 0 attracts {attr[0]:.0f} trips")
        """
        return np.sum(self.D, axis=0)
    
    def normalize_rows(self) -> None:
        """
        Normalize each row to sum to 1.0 (convert to probability matrix).
        
        Useful for creating destination choice probabilities from production totals.
        """
        row_sums = self.production()
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        self.D = self.D / row_sums[:, np.newaxis]
    
    def sparsity(self) -> float:
        """
        Calculate matrix sparsity (fraction of zero entries).
        
        Returns:
            Ratio of zero entries to total entries (0 to 1)
        
        Example:
            >>> sparsity = od.sparsity()
            >>> print(f"Matrix is {sparsity:.1%} sparse")
        """
        total_elements = self.n * self.n
        zero_count = np.sum(self.D == 0)
        return float(zero_count) / float(total_elements)
    
    def get_top_flows(self, k: int = 10) -> list[Tuple[int, int, float]]:
        """
        Get the k largest OD flows.
        
        Args:
            k: Number of top flows to return
        
        Returns:
            List of (origin, destination, flow) tuples, sorted descending
        
        Example:
            >>> top_flows = od.get_top_flows(k=5)
            >>> for i, j, flow in top_flows:
            ...     print(f"OD {i}→{j}: {flow:.0f} trips")
        """
        # Flatten matrix and get indices of top k values
        flat_indices = np.argsort(self.D.flatten())[::-1][:k]
        
        # Convert flat indices back to (i, j) pairs
        result = []
        for flat_idx in flat_indices:
            i = flat_idx // self.n
            j = flat_idx % self.n
            flow = self.D[i, j]
            if flow > 0:  # Only include positive flows
                result.append((int(i), int(j), float(flow)))
        
        return result
    
    def copy(self) -> ODDemand:
        """Create a copy of this OD demand matrix."""
        return ODDemand(self.D.copy())
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ODDemand({self.n}×{self.n}, "
            f"total={self.total_trips():.0f} trips, "
            f"sparsity={self.sparsity():.1%})"
        )


def gravity_od(
    masses: np.ndarray,
    distances: np.ndarray,
    alpha: float = 1.0,
    beta: float = 2.0,
    scale: float = 1.0,
    eps: float = 1e-9,
    zero_diag: bool = True,
) -> ODDemand:
    """
    Generate OD matrix using gravity model.
    
    The gravity model estimates trip flows between zones based on their
    "masses" (e.g., population, employment) and distances:
    
        T_ij = scale × (M_i^α × M_j^α) / (d_ij^β + ε)
    
    Args:
        masses: 1D array of zone masses (e.g., population)
        distances: n×n array of inter-zone distances
        alpha: Mass exponent (sensitivity to zone size)
        beta: Distance decay exponent (higher = stronger distance penalty)
        scale: Overall scaling factor for trip generation
        eps: Small constant to avoid division by zero
        zero_diag: Whether to zero out intra-zonal trips
    
    Returns:
        ODDemand object with computed trip matrix
    
    Raises:
        ValueError: If input dimensions are incompatible
    
    Example:
        >>> masses = np.array([1000, 2000, 1500])  # Population
        >>> distances = np.array([
        ...     [0, 5, 10],
        ...     [5, 0, 3],
        ...     [10, 3, 0]
        ... ])
        >>> od = gravity_od(masses, distances, alpha=1.0, beta=2.0, scale=0.01)
        >>> print(f"Total trips: {od.total_trips():.0f}")
    
    Note:
        - Higher beta values produce more localized trip patterns
        - Alpha typically ranges from 0.5 to 1.5
        - Beta typically ranges from 1.0 to 3.0
        - Use scale to match observed total trip counts
    """
    masses = np.asarray(masses, dtype=float)
    distances = np.asarray(distances, dtype=float)
    
    # Validation
    if masses.ndim != 1:
        raise ValueError(
            f"Masses must be 1D array, got {masses.ndim}D"
        )
    
    n = masses.size
    
    if distances.shape != (n, n):
        raise ValueError(
            f"Distance matrix must be ({n}, {n}), got {distances.shape}"
        )
    
    if alpha <= 0:
        logger.warning(f"alpha={alpha} is non-positive, which is unusual")
    
    if beta <= 0:
        logger.warning(f"beta={beta} is non-positive, distance decay will be inverted")
    
    # Reshape for broadcasting
    mass_origin = masses.reshape(n, 1)
    mass_dest = masses.reshape(1, n)
    
    # Gravity model formula
    numerator = (mass_origin ** alpha) * (mass_dest ** alpha)
    denominator = (distances ** beta) + eps
    
    trip_matrix = scale * (numerator / denominator)
    
    # Create OD demand object
    od = ODDemand(trip_matrix)
    
    if zero_diag:
        od.zero_diagonal()
    
    logger.info(
        f"Generated gravity OD matrix: {n}×{n}, "
        f"total trips={od.total_trips():.1f}, "
        f"sparsity={od.sparsity():.2%}"
    )
    
    return od


def normalize_to_total_trips(
    od: ODDemand,
    total_trips: float,
) -> ODDemand:
    """
    Scale OD matrix to match target total trip count.
    
    Args:
        od: Input OD demand matrix
        total_trips: Desired total trip count
    
    Returns:
        New ODDemand with scaled values
    
    Raises:
        ValueError: If total_trips is negative
    
    Example:
        >>> od = ODDemand(np.array([[0, 10], [20, 0]]))
        >>> scaled = normalize_to_total_trips(od, 60.0)
        >>> scaled.total_trips()
        60.0
    """
    if total_trips < 0:
        raise ValueError(
            f"Total trips must be non-negative, got {total_trips}"
        )
    
    current_total = od.total_trips()
    
    if current_total <= 0:
        logger.warning(
            "Cannot normalize OD matrix with zero total trips. "
            "Returning zero matrix."
        )
        return ODDemand(np.zeros_like(od.D))
    
    scaling_factor = total_trips / current_total
    scaled_matrix = od.D * scaling_factor
    
    logger.debug(
        f"Normalized OD matrix from {current_total:.1f} to {total_trips:.1f} trips "
        f"(factor={scaling_factor:.3f})"
    )
    
    return ODDemand(scaled_matrix)


def doubly_constrained_od(
    productions: np.ndarray,
    attractions: np.ndarray,
    distances: np.ndarray,
    beta: float = 2.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> ODDemand:
    """
    Generate OD matrix with doubly-constrained gravity model.
    
    Ensures row sums match production constraints and column sums
    match attraction constraints via iterative proportional fitting (IPF).
    
    Args:
        productions: Target trip productions from each zone
        attractions: Target trip attractions to each zone
        distances: Inter-zone distance matrix
        beta: Distance decay parameter
        max_iterations: Maximum IPF iterations
        tolerance: Convergence tolerance
    
    Returns:
        Balanced OD demand matrix
    
    Raises:
        ValueError: If productions and attractions don't sum to same value
    
    Example:
        >>> prod = np.array([100, 150, 200])  # Trips from each zone
        >>> attr = np.array([120, 180, 150])  # Trips to each zone
        >>> distances = np.array([[0, 5, 10], [5, 0, 3], [10, 3, 0]])
        >>> od = doubly_constrained_od(prod, attr, distances)
        >>> np.allclose(od.production(), prod)  # Check if balanced
        True
    
    Note:
        This is also known as the Furness method or bi-proportional fitting.
        Used when both trip productions and attractions are known/constrained.
    """
    productions = np.asarray(productions, dtype=float)
    attractions = np.asarray(attractions, dtype=float)
    distances = np.asarray(distances, dtype=float)
    
    n = len(productions)
    
    # Validation
    if len(attractions) != n or distances.shape != (n, n):
        raise ValueError("Dimension mismatch in inputs")
    
    prod_total = productions.sum()
    attr_total = attractions.sum()
    
    if not np.isclose(prod_total, attr_total, rtol=1e-3):
        raise ValueError(
            f"Productions ({prod_total:.1f}) and attractions ({attr_total:.1f}) "
            "must sum to same value"
        )
    
    # Initial estimate from simple gravity model
    cost_matrix = distances ** beta
    cost_matrix[cost_matrix == 0] = 1e-9
    T = np.ones((n, n)) / cost_matrix
    
    # Iterative proportional fitting (Furness method)
    for iteration in range(max_iterations):
        # Balance rows (productions)
        row_sums = T.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        T = T * (productions / row_sums)[:, np.newaxis]
        
        # Balance columns (attractions)
        col_sums = T.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        T = T * (attractions / col_sums)
        
        # Check convergence
        row_error = np.abs(T.sum(axis=1) - productions).max()
        col_error = np.abs(T.sum(axis=0) - attractions).max()
        
        if row_error < tolerance and col_error < tolerance:
            logger.info(
                f"Doubly-constrained OD converged in {iteration+1} iterations"
            )
            break
    else:
        logger.warning(
            f"Doubly-constrained OD did not converge after {max_iterations} iterations. "
            f"Row error: {row_error:.6f}, Col error: {col_error:.6f}"
        )
    
    return ODDemand(T)


def uniform_od(
    n: int,
    trips_per_pair: float = 10.0,
    zero_diag: bool = True,
) -> ODDemand:
    """
    Generate uniform OD matrix (all pairs have same demand).
    
    Useful for testing and baseline scenarios.
    
    Args:
        n: Number of zones
        trips_per_pair: Trips for each OD pair
        zero_diag: Whether to zero out diagonal
    
    Returns:
        ODDemand with uniform values
    
    Example:
        >>> od = uniform_od(n=5, trips_per_pair=20.0)
        >>> od.total_trips()
        400.0  # 5×5 - 5 diagonal = 20 pairs × 20 trips
    """
    D = np.full((n, n), trips_per_pair, dtype=float)
    
    od = ODDemand(D)
    if zero_diag:
        od.zero_diagonal()
    
    return od


def random_od(
    n: int,
    mean_trips: float = 50.0,
    std_trips: float = 20.0,
    seed: Optional[int] = None,
    zero_diag: bool = True,
) -> ODDemand:
    """
    Generate random OD matrix with normally distributed trips.
    
    Args:
        n: Number of zones
        mean_trips: Mean trips per OD pair
        std_trips: Standard deviation of trips
        seed: Random seed for reproducibility
        zero_diag: Whether to zero out diagonal
    
    Returns:
        ODDemand with random values (clipped to non-negative)
    
    Example:
        >>> od = random_od(n=10, mean_trips=50, std_trips=20, seed=42)
        >>> print(f"Total trips: {od.total_trips():.0f}")
    """
    rng = np.random.default_rng(seed)
    
    D = rng.normal(mean_trips, std_trips, size=(n, n))
    D = np.maximum(D, 0.0)  # Clip to non-negative
    
    od = ODDemand(D)
    if zero_diag:
        od.zero_diagonal()
    
    return od


def combine_od_matrices(
    od_list: list[ODDemand],
    weights: Optional[np.ndarray] = None,
) -> ODDemand:
    """
    Combine multiple OD matrices with optional weighting.
    
    Useful for aggregating time-period-specific matrices into daily totals,
    or combining different trip purposes.
    
    Args:
        od_list: List of ODDemand matrices (must be same size)
        weights: Optional weights for each matrix (default: equal weights)
    
    Returns:
        Combined ODDemand matrix
    
    Raises:
        ValueError: If matrices have different sizes
    
    Example:
        >>> am_peak = ODDemand(np.array([[0, 50], [30, 0]]))
        >>> pm_peak = ODDemand(np.array([[0, 40], [60, 0]]))
        >>> daily = combine_od_matrices([am_peak, pm_peak])
        >>> daily.total_trips()
        180.0
    """
    if not od_list:
        raise ValueError("od_list cannot be empty")
    
    n = od_list[0].n
    
    # Check all matrices are same size
    for i, od in enumerate(od_list):
        if od.n != n:
            raise ValueError(
                f"OD matrix {i} has size {od.n}×{od.n}, expected {n}×{n}"
            )
    
    # Set up weights
    if weights is None:
        weights = np.ones(len(od_list))
    else:
        weights = np.asarray(weights)
        if len(weights) != len(od_list):
            raise ValueError(
                f"weights length ({len(weights)}) doesn't match od_list ({len(od_list)})"
            )
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Combine
    combined = np.zeros((n, n), dtype=float)
    for od, w in zip(od_list, weights):
        combined += od.D * w
    
    return ODDemand(combined)


def calibrate_gravity_model(
    observed_od: ODDemand,
    masses: np.ndarray,
    distances: np.ndarray,
    alpha_range: Tuple[float, float] = (0.5, 1.5),
    beta_range: Tuple[float, float] = (1.0, 3.0),
    n_trials: int = 20,
) -> Tuple[float, float, float, ODDemand]:
    """
    Calibrate gravity model parameters to match observed OD matrix.
    
    Uses grid search to find best alpha, beta, and scale parameters.
    
    Args:
        observed_od: Observed/target OD matrix
        masses: Zone masses (population, employment, etc.)
        distances: Inter-zone distances
        alpha_range: Range for alpha parameter
        beta_range: Range for beta parameter
        n_trials: Number of trials per parameter
    
    Returns:
        Tuple of (best_alpha, best_beta, best_scale, best_od)
    
    Example:
        >>> # Calibrate to observed data
        >>> alpha, beta, scale, fitted_od = calibrate_gravity_model(
        ...     observed_od, masses, distances
        ... )
        >>> print(f"Best params: α={alpha:.2f}, β={beta:.2f}, scale={scale:.2f}")
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_trials)
    betas = np.linspace(beta_range[0], beta_range[1], n_trials)
    
    best_error = float('inf')
    best_params = (1.0, 2.0, 1.0)
    best_od = None
    
    target_total = observed_od.total_trips()
    
    for alpha in alphas:
        for beta in betas:
            # Generate OD with unit scale
            trial_od = gravity_od(masses, distances, alpha, beta, scale=1.0)
            
            # Compute required scale
            trial_total = trial_od.total_trips()
            if trial_total > 0:
                scale = target_total / trial_total
            else:
                continue
            
            # Rescale
            trial_od = normalize_to_total_trips(trial_od, target_total)
            
            # Compute error (RMSE)
            error = np.sqrt(np.mean((trial_od.D - observed_od.D) ** 2))
            
            if error < best_error:
                best_error = error
                best_params = (alpha, beta, scale)
                best_od = trial_od
    
    logger.info(
        f"Calibrated gravity model: α={best_params[0]:.3f}, "
        f"β={best_params[1]:.3f}, scale={best_params[2]:.3f}, RMSE={best_error:.2f}"
    )
    
    return best_params[0], best_params[1], best_params[2], best_od