"""Newsvendor optimal stock level and sensitivity analysis.

The classic newsvendor (single-period inventory) problem:
  A store must decide how much change (q) to prepare before the day starts.
  - If demand D > q: stockout cost C_u per unit short (customer walks away,
    cashier improvises, brand damage).
  - If demand D < q: overstock cost C_o per unit excess (capital tied up,
    security risk of holding cash overnight).

The optimal order quantity q* satisfies the critical fractile condition:

    q* = F⁻¹(C_u / (C_u + C_o))

where F is the CDF of demand and C_u / (C_u + C_o) is the critical ratio (CR).
This result follows directly from minimising the expected cost function and is
proved by taking the derivative and setting to zero.

For a continuous symmetric distribution this simplifies to the CR-th quantile
of the forecast distribution.  With posterior predictive samples from the
Bayesian model we estimate F empirically.

Reference: Arrow et al. (1951); Scarf (1958); Porteus (1990) Ch. 12 in
"Handbooks in OR & MS: Stochastic Models."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class NewsvendorResult:
    """Output of the newsvendor optimisation for one store-date.

    Attributes:
        store_id: Store identifier.
        date: Forecast date.
        q_star: Optimal cash buffer in MXN (critical-fractile quantity).
        critical_ratio: C_u / (C_u + C_o) — the target service level.
        expected_cost: Expected total cost at q* under the empirical distribution.
        q10: 10th-percentile demand (from posterior samples).
        q50: Median demand.
        q90: 90th-percentile demand.
    """
    store_id: str
    date: pd.Timestamp
    q_star: float
    critical_ratio: float
    expected_cost: float
    q10: float
    q50: float
    q90: float


class NewsvendorOptimizer:
    """Computes optimal daily cash buffers using the newsvendor critical fractile.

    Attributes:
        cost_underage: C_u — cost per MXN of unmet demand (stockout penalty).
        cost_overage: C_o — cost per MXN of excess cash held (overstock penalty).
        critical_ratio: Derived optimal service level = C_u / (C_u + C_o).
    """

    def __init__(self, cost_underage: float = 3.0, cost_overage: float = 1.0) -> None:
        if cost_underage <= 0 or cost_overage <= 0:
            raise ValueError("Costs must be strictly positive.")
        self.cost_underage = cost_underage
        self.cost_overage = cost_overage
        self.critical_ratio = cost_underage / (cost_underage + cost_overage)

    def optimise(
        self,
        store_id: str,
        date: pd.Timestamp,
        demand_samples: npt.NDArray[np.float64],
    ) -> NewsvendorResult:
        """
        Compute the optimal cash buffer for one store-date from demand samples.

        The critical-fractile formula gives the exact optimal q* for any demand
        distribution when costs are linear (proven by setting dE[Cost]/dq = 0):

            q* = F⁻¹(CR)  where CR = C_u / (C_u + C_o)

        We estimate F⁻¹ empirically from the posterior predictive samples.

        Args:
            store_id: Store identifier (passed through to output).
            date: Forecast date (passed through to output).
            demand_samples: 1-D array of demand draws from the posterior
                predictive distribution (MXN).

        Returns:
            NewsvendorResult with q*, expected cost, and demand quantiles.
        """
        samples = np.asarray(demand_samples, dtype=np.float64)
        q_star = float(np.quantile(samples, self.critical_ratio))
        expected_cost = self._expected_cost(samples, q_star)

        return NewsvendorResult(
            store_id=store_id,
            date=date,
            q_star=q_star,
            critical_ratio=self.critical_ratio,
            expected_cost=expected_cost,
            q10=float(np.quantile(samples, 0.10)),
            q50=float(np.quantile(samples, 0.50)),
            q90=float(np.quantile(samples, 0.90)),
        )

    def sensitivity(
        self,
        demand_samples: npt.NDArray[np.float64],
        cost_underage_range: tuple[float, float] = (1.0, 10.0),
        n_points: int = 20,
    ) -> pd.DataFrame:
        """
        Show how q* changes across a range of underage cost assumptions.

        Useful for presenting the model to business stakeholders: the analyst
        can show that q* is robust to moderate changes in the cost ratio, or
        identify the break-even cost at which the recommendation changes.

        Args:
            demand_samples: Posterior demand samples.
            cost_underage_range: (min, max) of C_u to sweep, with C_o fixed.
            n_points: Number of grid points for the sweep.

        Returns:
            DataFrame with columns [cost_underage, critical_ratio, q_star,
            expected_cost] sorted by cost_underage.
        """
        samples = np.asarray(demand_samples, dtype=np.float64)
        rows = []
        for c_u in np.linspace(*cost_underage_range, n_points):
            cr = c_u / (c_u + self.cost_overage)
            qs = float(np.quantile(samples, cr))
            rows.append({
                "cost_underage": round(c_u, 4),
                "critical_ratio": round(cr, 4),
                "q_star": round(qs, 2),
                "expected_cost": round(self._expected_cost(samples, qs), 2),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expected_cost(
        self,
        samples: npt.NDArray[np.float64],
        q: float,
    ) -> float:
        """
        Estimate expected total cost E[C_u·max(D-q,0) + C_o·max(q-D,0)] via MC.

        Args:
            samples: Demand draws.
            q: Stock level to evaluate.

        Returns:
            Monte Carlo estimate of expected cost.
        """
        underage = np.maximum(samples - q, 0.0)
        overage = np.maximum(q - samples, 0.0)
        return float(np.mean(self.cost_underage * underage + self.cost_overage * overage))
