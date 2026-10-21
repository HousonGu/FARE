"""Resolve planned Favorita holidays to the stores where they apply."""

from __future__ import annotations

import pandas as pd

def favorita_holiday_store_dates(stores: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    # Event includes unforeseen shocks (e.g. the 2016 earthquake); Work Day is
    # not a holiday. A transferred original date is not the observed holiday.
    planned = holidays.loc[
        (~holidays["transferred"]) & holidays["type"].isin(
            ["Holiday", "Transfer", "Additional", "Bridge"]
        )
    ]
    national = stores[["store_nbr"]].merge(
        planned.loc[planned["locale"] == "National", ["date"]], how="cross"
    )
    regional = stores[["store_nbr", "state"]].merge(
        planned.loc[planned["locale"] == "Regional", ["date", "locale_name"]],
        left_on="state", right_on="locale_name"
    )[["store_nbr", "date"]]
    local = stores[["store_nbr", "city"]].merge(
        planned.loc[planned["locale"] == "Local", ["date", "locale_name"]],
        left_on="city", right_on="locale_name"
    )[["store_nbr", "date"]]
    return pd.concat([national, regional, local], ignore_index=True).drop_duplicates()
