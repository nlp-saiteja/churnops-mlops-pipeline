"""
Simulates production data arriving over time — with artificial drift injected.

In real life, this would be actual customer records coming in from a database.
Here we simulate two scenarios:
  - No drift:   new data looks like training data
  - With drift: new data has shifted distributions (more Fiber optic users,
                shorter tenures, higher monthly charges)
"""

import pandas as pd
import numpy as np
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def simulate_production_data(drift: bool = True, n_samples: int = 500, seed: int = 99) -> pd.DataFrame:
    """
    Creates a fake batch of production data based on the processed training set.
    If drift=True, shifts key feature distributions to simulate real-world change.
    """
    config = load_config()
    processed_path = config["data"]["processed_path"]

    rng = np.random.default_rng(seed)

    # Load training data as the base
    X_train = pd.read_csv(f"{processed_path}/X_train.csv")

    # Sample from training data
    sample = X_train.sample(n=n_samples, replace=True, random_state=seed).copy()

    if drift:
        print("Injecting drift into production data...")

        # Drift 1: Customers are newer (shorter tenure)
        # Original mean ~32 months → drifted mean ~12 months
        sample["tenure"] = np.clip(
            rng.normal(loc=12, scale=8, size=n_samples).astype(int), 0, 72
        )

        # Drift 2: Higher monthly charges (price increases)
        # Original mean ~$65 → drifted mean ~$85
        sample["MonthlyCharges"] = np.clip(
            rng.normal(loc=85, scale=20, size=n_samples), 20, 120
        )

        # Drift 3: More Fiber optic users (market shift)
        # Original ~44% → drifted ~75%
        sample["InternetService_Fiber optic"] = rng.binomial(1, 0.75, n_samples).astype(bool)
        sample["InternetService_No"] = rng.binomial(1, 0.05, n_samples).astype(bool)

        # Drift 4: More month-to-month contracts
        # Original ~55% → drifted ~80%
        sample["Contract_One year"] = rng.binomial(1, 0.10, n_samples).astype(bool)
        sample["Contract_Two year"] = rng.binomial(1, 0.10, n_samples).astype(bool)

        # Recalculate TotalCharges
        sample["TotalCharges"] = sample["tenure"] * sample["MonthlyCharges"]

    else:
        print("Generating stable production data (no drift)...")

    return sample


if __name__ == "__main__":
    drifted = simulate_production_data(drift=True)
    stable = simulate_production_data(drift=False)
    print(f"\nDrifted data — mean tenure: {drifted['tenure'].mean():.1f}, "
          f"mean MonthlyCharges: {drifted['MonthlyCharges'].mean():.1f}")
    print(f"Stable data  — mean tenure: {stable['tenure'].mean():.1f}, "
          f"mean MonthlyCharges: {stable['MonthlyCharges'].mean():.1f}")
