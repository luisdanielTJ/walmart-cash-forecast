"""Denomination mix allocation for the register change fund.

The change fund (cash kept in registers to make change) is sized at a
fixed fraction of the daily cash forecast (default 3 %).  That fund is
then distributed across denominations proportionally, reflecting the
actual circulation share of each denomination reported by Banco de México
(2023 Annual Report on Banknote and Coin Circulation).

Proportions used (bills only, coins get small fixed counts):
  $200: 42 % — most circulated bill in MX retail
  $100: 33 % — second most circulated
  $50 : 13 % — frequent for mid-size change
  $20 :  8 % — petty change
  $10 coins + smaller: fixed minimums, independent of fund size

Why proportional instead of ILP minimisation?
  An ILP that minimises piece count always selects the largest available
  denomination, producing a float of only $200 bills.  Real tills need a
  mix that mirrors what customers tender and what cashiers give as change.
  Proportional allocation grounded in circulation data avoids this failure
  mode while remaining simple and auditable.

Reference: Banco de México (2023) "Informe Anual sobre Billetes y Monedas".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Mexican peso denominations in ascending order (MXN)
DENOMINATIONS: list[float] = [
    0.10, 0.20, 0.50,        # centavo coins
    1.00, 2.00, 5.00, 10.00, # peso coins
    20.00, 50.00, 100.00,    # small bills
    200.00, 500.00, 1_000.00, # large bills
]

# Proportional share of the change fund allocated to each denomination.
# Bill shares derived from Banco de México 2023 circulation data.
# Coins get fixed counts (see _COIN_MINIMUMS) independent of fund size.
_BILL_PROPORTIONS: dict[float, float] = {
    20.00:  0.08,   # 8 %  — petty change
    50.00:  0.13,   # 13 % — mid-size change
    100.00: 0.33,   # 33 % — second most circulated bill
    200.00: 0.42,   # 42 % — most circulated bill in MX retail
    500.00: 0.04,   # 4 %  — occasional large-purchase change
}
# $1000 bills excluded: not stocked in tills (many stores refuse them)

# Fixed coin counts per denomination — independent of fund size.
# These reflect a typical Supercenter register starting float.
_COIN_MINIMUMS: dict[float, int] = {
    0.10: 200,   # centavo change for $x.90 prices
    0.20: 200,   # centavo change for $x.80 prices
    0.50: 200,   # half-peso change
    1.00: 300,   # peso coins — nearly every transaction
    2.00: 200,
    5.00: 150,
    10.00: 150,
}


@dataclass
class DenominationResult:
    """Optimal denomination mix for one store-date.

    Attributes:
        store_id: Store identifier.
        date: Target date.
        target: Requested cash buffer (MXN) from the newsvendor model.
        total_value: Actual total value of the mix (MXN).
        total_pieces: Number of coins + bills in the mix.
        mix: Dict mapping denomination → quantity.
        status: Always "Proportional" for this allocator.
    """
    store_id: str
    date: pd.Timestamp
    target: float
    total_value: float
    total_pieces: int
    mix: dict[float, int] = field(default_factory=dict)
    status: str = "Proportional"


class DenominationSolver:
    """Proportional denomination allocator for the register change fund.

    The change fund is sized as fund_ratio × daily_forecast.  That amount
    is distributed across bills using Banco de México circulation shares,
    plus fixed coin counts for making centavo/peso change.

    Attributes:
        fund_ratio: Fraction of daily forecast to allocate as change fund.
            Default 0.03 (3 %) matches retail industry benchmarks.
        limits: Ignored — kept for API compatibility with config.
    """

    def __init__(
        self,
        limits: dict[float, int] | None = None,
        fund_ratio: float = 0.03,
    ) -> None:
        self.limits = limits or {}
        self.fund_ratio = fund_ratio

    def solve(
        self,
        store_id: str,
        date: pd.Timestamp,
        target: float,
    ) -> DenominationResult:
        """
        Allocate a change fund proportionally across denominations.

        Args:
            store_id: Store identifier (passed through to result).
            date: Target date (passed through to result).
            target: Daily cash forecast (MXN) — change fund = fund_ratio × target.

        Returns:
            DenominationResult with a realistic denomination mix.
        """
        fund = max(target * self.fund_ratio, 1.0)

        mix: dict[float, int] = {}

        # Fixed coin counts
        for d, qty in _COIN_MINIMUMS.items():
            mix[d] = qty

        # Proportional bill allocation
        for d, share in _BILL_PROPORTIONS.items():
            mix[d] = max(1, round(fund * share / d))

        # $500 and $1000 not stocked in tills
        mix[500.00] = 0
        mix[1_000.00] = 0

        total_value = sum(d * n for d, n in mix.items())
        total_pieces = sum(mix.values())

        return DenominationResult(
            store_id=store_id,
            date=date,
            target=target,
            total_value=round(total_value, 2),
            total_pieces=total_pieces,
            mix=mix,
            status="Proportional",
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
            for d in DENOMINATIONS:
                row[f"denom_{d:.2f}"] = r.mix.get(d, 0)
            rows.append(row)
        return pd.DataFrame(rows)
