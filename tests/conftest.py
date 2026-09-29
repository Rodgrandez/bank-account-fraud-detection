import numpy as np
import pandas as pd


def make_baf(n=8000, seed=0):
    """Small synthetic frame with the BAF schema subset used by the package (not real data)."""
    rng = np.random.default_rng(seed)
    month = rng.integers(0, 8, n)
    age = rng.choice([20, 30, 40, 50, 60, 70], n, p=[0.2, 0.25, 0.2, 0.15, 0.12, 0.08])
    os_ = rng.choice(["windows", "linux", "macintosh", "other"], n)
    velocity = rng.normal(5000, 1500, n)
    logit = -3.6 + 1.2 * (os_ == "windows") + 0.0004 * (velocity - 5000) + 0.02 * (age - 40) + 0.08 * month
    fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "fraud_bool": fraud, "month": month, "customer_age": age,
        "income": rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], n),
        "velocity_6h": velocity,
        "prev_address_months_count": rng.integers(-1, 200, n),
        "current_address_months_count": rng.integers(-1, 300, n),
        "bank_months_count": rng.integers(-1, 32, n),
        "session_length_in_minutes": np.where(rng.random(n) < 0.02, -1.0, rng.exponential(8, n)),
        "intended_balcon_amount": rng.normal(0, 20, n),
        "device_distinct_emails_8w": rng.integers(-1, 3, n),
        "payment_type": rng.choice(["AA", "AB", "AC", "AD"], n),
        "employment_status": rng.choice(["CA", "CB", "CC"], n),
        "housing_status": rng.choice(["BA", "BB", "BC"], n),
        "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[0.99, 0.01]),
        "device_os": os_,
        "device_fraud_count": 0,
    })
