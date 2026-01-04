from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenConfig:
    seed: int = 42
    n_patients: int = 50_000
    start_date: str = "2024-01-01"
    months: int = 24 


CFG = GenConfig()


# ----------------------------
# Helpers
# ----------------------------
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _month_start_dates(start_date: str, months: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_date).normalize()
    return pd.date_range(start=start, periods=months, freq="MS")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _choice(r: np.random.Generator, items: list[str], p: list[float] | None = None, size: int = 1) -> np.ndarray:
    return r.choice(items, size=size, replace=True, p=p)


# ----------------------------
# Data generation
# ----------------------------
def generate_patients(r: np.random.Generator, n: int) -> pd.DataFrame:
    patient_ids = np.arange(1, n + 1)

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_p = [0.12, 0.22, 0.22, 0.18, 0.16, 0.10]

    market_types = ["developed", "emerging"]
    market_p = [0.65, 0.35]

    # Simple country mapping
    countries_dev = ["US", "DE", "FR", "UK", "JP", "CA", "AU", "NL", "SE", "CH"]
    countries_em = ["BR", "MX", "PL", "TR", "IN", "ID", "ZA", "EG", "PH", "VN"]

    market = _choice(r, market_types, market_p, size=n)
    country = np.where(
        market == "developed",
        _choice(r, countries_dev, size=n),
        _choice(r, countries_em, size=n),
    )

    income_segments = ["low", "mid", "high"]
    # developed tend to higher income segment
    income = np.where(
        market == "developed",
        _choice(r, income_segments, p=[0.20, 0.55, 0.25], size=n),
        _choice(r, income_segments, p=[0.45, 0.45, 0.10], size=n),
    )

    conditions = ["myopia", "astigmatism", "presbyopia", "dry_eye"]
    cond_p = [0.45, 0.25, 0.20, 0.10]
    condition = _choice(r, conditions, cond_p, size=n)

    return pd.DataFrame({
        "patient_id": patient_ids,
        "age_group": _choice(r, age_groups, age_p, size=n),
        "country": country,
        "market_type": market,
        "income_segment": income,
        "vision_condition": condition,
    })


def generate_visits(r: np.random.Generator, patients: pd.DataFrame) -> pd.DataFrame:
    # Each patient: 1 diagnosis visit; optional fitting and follow-ups
    n = len(patients)
    months = _month_start_dates(CFG.start_date, CFG.months)

    # Base probabilities depend on market and condition
    is_dev = (patients["market_type"].values == "developed")
    has_dry = (patients["vision_condition"].values == "dry_eye")

    # Visit counts
    fitting_prob = np.where(is_dev, 0.65, 0.45)
    follow_prob = np.where(is_dev, 0.55, 0.35)
    follow_prob = np.where(has_dry, follow_prob + 0.10, follow_prob)

    # Channels
    channels = ["clinic", "retail", "ecomm"]
    # clinic dominates diagnosis; others appear later
    diag_channel = _choice(r, ["clinic", "clinic", "clinic", "retail"], size=n)  # biased to clinic

    # Clinic IDs: 200 clinics
    clinic_ids = r.integers(1, 201, size=n)

    diag_month = r.integers(0, len(months), size=n)
    diag_date = months[diag_month] + pd.to_timedelta(r.integers(0, 28, size=n), unit="D")

    visits = []
    visit_id = 1

    for i in range(n):
        pid = int(patients.iloc[i]["patient_id"])

        # diagnosis
        visits.append({
            "visit_id": visit_id,
            "patient_id": pid,
            "visit_date": diag_date[i],
            "visit_type": "diagnosis",
            "clinic_id": int(clinic_ids[i]),
            "channel": str(diag_channel[i]),
        })
        visit_id += 1

        # fitting (if contact lens related)
        if r.random() < fitting_prob[i]:
            vdate = diag_date[i] + pd.to_timedelta(int(r.integers(3, 21)), unit="D")
            visits.append({
                "visit_id": visit_id,
                "patient_id": pid,
                "visit_date": vdate,
                "visit_type": "fitting",
                "clinic_id": int(clinic_ids[i]),
                "channel": "clinic",
            })
            visit_id += 1

        # follow-ups: 0–3
        if r.random() < follow_prob[i]:
            n_follow = int(r.integers(1, 4))
            last = diag_date[i]
            for _ in range(n_follow):
                last = last + pd.to_timedelta(int(r.integers(20, 70)), unit="D")
                visits.append({
                    "visit_id": visit_id,
                    "patient_id": pid,
                    "visit_date": last,
                    "visit_type": "follow_up",
                    "clinic_id": int(clinic_ids[i]),
                    "channel": str(_choice(r, channels, p=[0.50, 0.30, 0.20], size=1)[0]),
                })
                visit_id += 1

    visits_df = pd.DataFrame(visits)
    visits_df["visit_date"] = pd.to_datetime(visits_df["visit_date"])
    return visits_df.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)


def generate_recommendations(r: np.random.Generator, patients: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    # Use diagnosis date as recommendation date (per patient)
    first_diag = (
        visits[visits["visit_type"] == "diagnosis"]
        .sort_values(["patient_id", "visit_date"])
        .groupby("patient_id", as_index=False)
        .first()[["patient_id", "visit_date"]]
        .rename(columns={"visit_date": "rec_date"})
    )

    # Not everyone gets product recommendation
    is_dev = (patients.set_index("patient_id").loc[first_diag["patient_id"], "market_type"].values == "developed")
    base_rec_prob = np.where(is_dev, 0.80, 0.65)
    rec_mask = r.random(len(first_diag)) < base_rec_prob
    rec_base = first_diag[rec_mask].copy()

    # Categories
    categories = ["contacts", "solution", "drops"]
    # Dry eye -> more drops
    cond = patients.set_index("patient_id").loc[rec_base["patient_id"], "vision_condition"].values
    cat = []
    for c in cond:
        if c == "dry_eye":
            cat.append(_choice(r, categories, p=[0.25, 0.20, 0.55], size=1)[0])
        else:
            cat.append(_choice(r, categories, p=[0.55, 0.30, 0.15], size=1)[0])

    # Tier depends on market & income
    income = patients.set_index("patient_id").loc[rec_base["patient_id"], "income_segment"].values
    market = patients.set_index("patient_id").loc[rec_base["patient_id"], "market_type"].values

    tiers = []
    for m, inc in zip(market, income):
        if m == "developed":
            p = [0.20, 0.50, 0.30] if inc != "low" else [0.35, 0.50, 0.15]
        else:
            p = [0.45, 0.45, 0.10] if inc != "high" else [0.30, 0.55, 0.15]
        tiers.append(_choice(r, ["value", "mid", "premium"], p=p, size=1)[0])

    # Placeholder brands
    brands = ["OptiClear", "VisionPro", "AquaLens", "BrightSight", "PureCare"]
    wear = ["daily", "monthly"]

    rec_base = rec_base.reset_index(drop=True)
    rec_base.insert(0, "rec_id", np.arange(1, len(rec_base) + 1))
    rec_base["category"] = cat
    rec_base["product_tier"] = tiers
    rec_base["recommended_brand"] = _choice(r, brands, size=len(rec_base))
    rec_base["expected_wear"] = np.where(rec_base["category"].eq("contacts"),
                                         _choice(r, wear, p=[0.55, 0.45], size=len(rec_base)),
                                         "")
    rec_base["rec_date"] = pd.to_datetime(rec_base["rec_date"])
    return rec_base


def generate_transactions(r: np.random.Generator, patients: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    # Adoption depends on market/channel/tier; generate purchases over time
    # Prices by category & tier
    base_price = {
        ("contacts", "value"): 18,
        ("contacts", "mid"): 28,
        ("contacts", "premium"): 40,
        ("solution", "value"): 6,
        ("solution", "mid"): 9,
        ("solution", "premium"): 12,
        ("drops", "value"): 8,
        ("drops", "mid"): 12,
        ("drops", "premium"): 18,
    }

    patient_map = patients.set_index("patient_id")
    tx_rows = []
    tx_id = 1

    for _, row in recs.iterrows():
        pid = int(row["patient_id"])
        rec_date = pd.Timestamp(row["rec_date"])
        category = str(row["category"])
        tier = str(row["product_tier"])

        market = str(patient_map.loc[pid, "market_type"])
        income = str(patient_map.loc[pid, "income_segment"])

        # adoption probability
        p_adopt = 0.70 if market == "developed" else 0.55
        if tier == "premium":
            p_adopt -= 0.10 if income in ["low", "mid"] else 0.02
        if category == "drops":
            p_adopt += 0.05  # easier to adopt

        if r.random() > p_adopt:
            continue  # no purchase at all

        # number of transactions (repeat purchases)
        # contacts -> more repeats; solutions moderate; drops moderate
        if category == "contacts":
            n_tx = int(r.integers(2, 6))
            gap_days = (25, 45)
        elif category == "solution":
            n_tx = int(r.integers(1, 5))
            gap_days = (20, 50)
        else:
            n_tx = int(r.integers(1, 4))
            gap_days = (15, 60)

        # Channel for purchase
        channel = _choice(r, ["retail", "ecomm", "clinic"], p=[0.55, 0.30, 0.15], size=1)[0]
        if market == "developed":
            # a bit more ecomm
            channel = _choice(r, ["retail", "ecomm", "clinic"], p=[0.45, 0.45, 0.10], size=1)[0]

        last_date = rec_date + pd.to_timedelta(int(r.integers(0, 14)), unit="D")
        for _ in range(n_tx):
            price = base_price[(category, tier)]
            # small random variation
            price = float(max(1, price + r.normal(0, price * 0.08)))
            qty = int(r.integers(1, 3))
            tx_rows.append({
                "tx_id": tx_id,
                "patient_id": pid,
                "tx_date": last_date,
                "category": category,
                "product_tier": tier,
                "price": round(price, 2),
                "quantity": qty,
                "channel": str(channel),
            })
            tx_id += 1
            last_date = last_date + pd.to_timedelta(int(r.integers(gap_days[0], gap_days[1])), unit="D")

    tx = pd.DataFrame(tx_rows)
    if len(tx) == 0:
        return tx

    tx["tx_date"] = pd.to_datetime(tx["tx_date"])
    return tx.sort_values(["patient_id", "tx_date"]).reset_index(drop=True)


def generate_outcomes(r: np.random.Generator, patients: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    # One outcome record per patient at last observed month (or fixed horizon)
    patient_map = patients.set_index("patient_id")

    if len(tx) == 0:
        # fallback: create empty outcomes
        return pd.DataFrame(columns=["patient_id", "date", "satisfaction_score", "symptom_improved", "dropout_flag"])

    last_tx = tx.groupby("patient_id", as_index=False)["tx_date"].max()

    rows = []
    for _, row in last_tx.iterrows():
        pid = int(row["patient_id"])
        date = pd.Timestamp(row["tx_date"]) + pd.to_timedelta(int(r.integers(7, 30)), unit="D")

        market = str(patient_map.loc[pid, "market_type"])
        cond = str(patient_map.loc[pid, "vision_condition"])

        # satisfaction: developed slightly higher; dry eye slightly lower
        base = 6.8 if market == "developed" else 6.2
        if cond == "dry_eye":
            base -= 0.6

        sat = float(np.clip(base + r.normal(0, 1.2), 1, 10))
        improved = int(r.random() < (0.75 if sat >= 6 else 0.45))

        # dropout probability inversely related to satisfaction
        p_drop = 0.35 if market == "emerging" else 0.25
        p_drop += 0.10 if cond == "dry_eye" else 0.0
        p_drop += 0.15 if sat < 5.5 else 0.0
        dropout = int(r.random() < p_drop)

        rows.append({
            "patient_id": pid,
            "date": date,
            "satisfaction_score": round(sat, 1),
            "symptom_improved": improved,
            "dropout_flag": dropout,
        })

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out


def save_all(out_dir: Path, patients: pd.DataFrame, visits: pd.DataFrame, recs: pd.DataFrame,
             tx: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    _ensure_dir(out_dir)

    paths = {
        "patients": out_dir / "patients.csv",
        "visits": out_dir / "visits.csv",
        "recommendations": out_dir / "recommendations.csv",
        "transactions": out_dir / "transactions.csv",
        "outcomes": out_dir / "outcomes.csv",
    }

    patients.to_csv(paths["patients"], index=False)
    visits.to_csv(paths["visits"], index=False)
    recs.to_csv(paths["recommendations"], index=False)
    tx.to_csv(paths["transactions"], index=False)
    outcomes.to_csv(paths["outcomes"], index=False)

    print("\n✅ Synthetic datasets created:")
    for k, p in paths.items():
        nrows = {"patients": len(patients), "visits": len(visits), "recommendations": len(recs),
                 "transactions": len(tx), "outcomes": len(outcomes)}[k]
        print(f" - {p.as_posix()}  ({nrows:,} rows)")


def main() -> None:
    r = _rng(CFG.seed)

    project_root = Path(__file__).resolve().parents[1] 
    out_dir = project_root / "data" / "raw"

    patients = generate_patients(r, CFG.n_patients)
    visits = generate_visits(r, patients)
    recs = generate_recommendations(r, patients, visits)
    tx = generate_transactions(r, patients, recs)
    outcomes = generate_outcomes(r, patients, tx)

    save_all(out_dir, patients, visits, recs, tx, outcomes)


if __name__ == "__main__":
    main()
