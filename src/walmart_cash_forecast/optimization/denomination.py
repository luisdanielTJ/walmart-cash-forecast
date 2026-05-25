"""Integer Linear Programme (ILP) for optimal denomination mix.

Given a cash buffer target T (from the newsvendor optimiser), the store must
decide how many of each denomination to hold.  The ILP minimises total pieces
(coins + bills) while guaranteeing the total value covers T and respects
per-denomination capacity limits.

Problem formulation:
    min  Σ_d  x_d                       (minimise piece count)
    s.t. Σ_d  v_d · x_d  ≥  T           (meet cash buffer target)
         0 ≤ x_d ≤ L_d  ∀ d             (capacity per denomination)
         x_d ∈ ℤ⁺                       (integer quantities)

where v_d = denomination value in MXN, L_d = max units of denomination d.
Minimising pieces is operationally sensible: fewer coins/bills to count,
sort, and transport, and lower security risk.

Denominations (Banco de México):
  Coins:  $0.10, $0.20, $0.50, $1, $2, $5, $10
  Bills:  $20, $50, $100, $200, $500, $1,000

The CBC solver (bundled with PuLP) is used; the ILP is small enough that
branch-and-bound converges in milliseconds.

Reference: Cornuéjols & Tütüncü (2006) "Optimization Methods in Finance",
Ch. 4 (Integer Programming).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

# Mexican peso denominations in ascending order (MXN)
DENOMINATIONS: list[float] = [
    0.10, 0.20, 0.50,        # centavo coins
    1.00, 2.00, 5.00, 10.00, # peso coins
    20.00, 50.00, 100.00,    # small bills
    200.00, 500.00, 1_000.00, # large bills
]

# Default max units per denomination when not specified in config.
# Limits are set conservatively for a mid-size Bodega format; Supercenter
# limits are typically higher and should be overridden via config.
_DEFAULT_LIMITS: dict[float, int] = {
    0.10:     500,
    0.20:     500,
    0.50:     300,
    1.00:     300,
    2.00:     200,
    5.00:     200,
    10.00:    200,
    20.00:    200,
    50.00:    200,
    100.00:   200,
    200.00:   200,
    500.00:   200,
    1_000.00: 200,
}


@dataclass
class DenominationResult:
    """Optimal denomination mix for one store-date.

    Attributes:
        store_id: Store identifier.
        date: Target date.
        target: Requested cash buffer (MXN) from the newsvendor model.
        total_value: Actual total value of the mix (≥ target).
        total_pieces: Number of coins + bills in the mix.
        mix: Dict mapping denomination → quantity.
        status: PuLP solver status string (e.g. "Optimal").
    """
    store_id: str
    date: pd.Timestamp
    target: float
    total_value: float
    total_pieces: int
    mix: dict[float, int] = field(default_factory=dict)
    status: str = "Optimal"


class DenominationSolver:
    """ILP-based denomination mix optimiser using PuLP + CBC.

    Attributes:
        limits: Per-denomination maximum unit counts.  Defaults to
            _DEFAULT_LIMITS; can be overridden per store format via config.
    """

    def __init__(self, limits: dict[float, int] | None = None) -> None:
        self.limits: dict[float, int] = limits if limits is not None else dict(_DEFAULT_LIMITS)

    def solve(
        self,
        store_id: str,
        date: pd.Timestamp,
        target: float,
    ) -> DenominationResult:
        """
        Find the minimum-pieces denomination mix covering target MXN.

        Args:
            store_id: Store identifier (passed through to result).
            date: Target date (passed through to result).
            target: Cash buffer in MXN to cover (from newsvendor q*).

        Returns:
            DenominationResult with the optimal mix, or the best feasible
            solution if the ILP cannot be solved to optimality.

        Raises:
            RuntimeError: If the solver returns Infeasible (capacity limits are
                too tight to cover the target).
        """
        prob = pulp.LpProblem("denomination_mix", pulp.LpMinimize)

        # Decision variables: integer count for each denomination
        vars_: dict[float, pulp.LpVariable] = {
            d: pulp.LpVariable(
                f"x_{int(d * 100):05d}",   # e.g. x_00010 for $0.10
                lowBound=0,
                upBound=self.limits.get(d, _DEFAULT_LIMITS.get(d, 1000)),
                cat="Integer",
            )
            for d in DENOMINATIONS
        }

        # Objective: minimise total piece count
        prob += pulp.lpSum(vars_.values()), "total_pieces"

        # Constraint: total value must cover the buffer target
        prob += (
            pulp.lpSum(d * vars_[d] for d in DENOMINATIONS) >= target,
            "cover_target",
        )

        # Suppress PuLP/CBC console output
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status == "Infeasible":
            raise RuntimeError(
                f"ILP infeasible for store={store_id}, target={target:.2f} MXN. "
                "Increase denomination limits in config."
            )

        mix = {d: int(v.varValue or 0) for d, v in vars_.items() if (v.varValue or 0) > 0}
        total_value = sum(d * n for d, n in mix.items())
        total_pieces = sum(mix.values())

        return DenominationResult(
            store_id=store_id,
            date=date,
            target=target,
            total_value=round(total_value, 2),
            total_pieces=total_pieces,
            mix=mix,
            status=status,
        )

    def solve_batch(
        self,
        targets: pd.DataFrame,
    ) -> list[DenominationResult]:
        """
        Solve denomination mix for multiple store-dates.

        Args:
            targets: DataFrame with columns store_id, date, q_star.

        Returns:
            List of DenominationResult, one per row in targets.
        """
        results = []
        for _, row in targets.iterrows():
            result = self.solve(
                store_id=str(row["store_id"]),
                date=pd.Timestamp(row["date"]),
                target=float(row["q_star"]),
            )
            results.append(result)
        return results

    def to_dataframe(self, results: list[DenominationResult]) -> pd.DataFrame:
        """Convert a list of DenominationResult to a tidy wide DataFrame."""
        rows = []
        for r in results:
            row: dict = {
                "store_id": r.store_id,
                "date": r.date,
                "target": r.target,
                "total_value": r.total_value,
                "total_pieces": r.total_pieces,
                "status": r.status,
            }
            # Add one column per denomination (zero if not in mix)
            for d in DENOMINATIONS:
                row[f"denom_{d:.2f}"] = r.mix.get(d, 0)
            rows.append(row)
        return pd.DataFrame(rows)
