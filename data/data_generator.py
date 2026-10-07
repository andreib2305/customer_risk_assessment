import numpy as np
import pandas as pd
from models.fuzzy_logic import calculate_fuzzy_risk


def generate_dataset(n_samples=1500, random_state=42):
    np.random.seed(random_state)

    age = np.random.randint(18, 70, n_samples)
    income = np.random.randint(20000, 200000, n_samples)
    months_with_company = np.random.randint(1, 72, n_samples)
    complaints = np.random.poisson(lam=0.8, size=n_samples)
    service_score = np.random.randint(1, 11, n_samples)
    payment_delay_days = np.random.choice([0, 0, 0, 5, 10, 15, 30], n_samples)
    tariff = np.random.choice(['standard', 'premium'], n_samples, p=[0.7, 0.3])
    channel = np.random.choice(['web', 'office', 'mobile'], n_samples, p=[0.5, 0.2, 0.3])

    churn_prob = (
            0.05 * (payment_delay_days > 5) +
            0.08 * (complaints > 1) +
            0.04 * (service_score < 5) +
            0.03 * (months_with_company < 6) -
            0.02 * (tariff == 'premium')
    )
    churn_prob = np.clip(churn_prob + np.random.uniform(0, 0.2, n_samples), 0, 1)
    churn = (churn_prob > 0.22).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'months_with_company': months_with_company,
        'complaints': complaints,
        'service_score': service_score,
        'payment_delay_days': payment_delay_days,
        'tariff': tariff,
        'channel': channel,
        'churn': churn
    })

    df['complaints'] = df['complaints'].fillna(df['complaints'].median())
    df['service_score'] = df['service_score'].fillna(df['service_score'].median())
    df['fuzzy_risk_index'] = df.apply(calculate_fuzzy_risk, axis=1)

    return df