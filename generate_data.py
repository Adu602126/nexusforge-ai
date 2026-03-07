"""
============================================================
NexusForge — Dummy Data Generator
============================================================
Generates 100 realistic BFSI customer records.
Saves to data/customers.csv for campaign testing.
Run once: python data/generate_data.py
============================================================
"""

import os
import numpy as np
import pandas as pd

def generate_customers(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate n fake BFSI customer records."""
    np.random.seed(seed)

    # ── Name Components ─────────────────────────────────
    first_names = [
        "Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Arjun", "Deepa",
        "Karan", "Nisha", "Rohan", "Pooja", "Amit", "Kavya", "Siddharth", "Meera",
        "Rajesh", "Sunita", "Aditya", "Geeta", "Nikhil", "Swati", "Tushar", "Rekha",
        "Varun", "Smita", "Harsh", "Divya", "Ankur", "Neha", "Gaurav", "Ritu",
        "Manish", "Asha", "Sandeep", "Preeti", "Vivek", "Shilpa", "Pranav", "Usha",
    ]
    last_names = [
        "Sharma", "Patel", "Singh", "Gupta", "Kumar", "Mehta", "Verma", "Joshi",
        "Nair", "Reddy", "Iyer", "Pillai", "Das", "Bose", "Roy", "Ghosh",
        "Malhotra", "Kapoor", "Saxena", "Trivedi", "Pandey", "Mishra", "Agarwal", "Sinha",
    ]

    cities = [
        "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata",
        "Ahmedabad", "Jaipur", "Surat", "Lucknow", "Chandigarh", "Bhopal", "Kochi"
    ]

    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "rediffmail.com"]

    # Generate records
    records = []
    for i in range(n):
        fname = np.random.choice(first_names)
        lname = np.random.choice(last_names)
        name = f"{fname} {lname}"
        city = np.random.choice(cities)

        # Income: log-normal distribution centered around ₹6L/year
        income = int(np.random.lognormal(mean=13.3, sigma=0.6))  # ~₹300k-₹2M range
        income = max(100000, min(income, 5000000))  # clip to realistic range

        # Email
        email = f"{fname.lower()}.{lname.lower()}{np.random.randint(1, 99)}@{np.random.choice(domains)}"

        # Past opens (Poisson distribution, avg 3.5)
        past_opens = int(np.random.poisson(3.5))

        # Age (25-65)
        age = int(np.random.normal(38, 10))
        age = max(25, min(age, 65))

        # Product affinity based on income
        if income > 1500000:
            product_interest = np.random.choice(["investment", "credit_card", "fd_offer"], p=[0.5, 0.3, 0.2])
        elif income > 500000:
            product_interest = np.random.choice(["loan_offer", "investment", "insurance"], p=[0.4, 0.3, 0.3])
        else:
            product_interest = np.random.choice(["loan_offer", "insurance", "fd_offer"], p=[0.5, 0.3, 0.2])

        # CIBIL score
        cibil = int(np.random.normal(720, 60))
        cibil = max(300, min(cibil, 900))

        records.append({
            "id": i + 1,
            "name": name,
            "email": email,
            "age": age,
            "income": income,
            "location": city,
            "past_opens": past_opens,
            "product_interest": product_interest,
            "cibil_score": cibil,
            "has_existing_account": np.random.choice([True, False], p=[0.6, 0.4]),
            "preferred_language": np.random.choice(["English", "Hindi", "Regional"], p=[0.5, 0.35, 0.15]),
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_customers(100)
    df.to_csv("data/customers.csv", index=False)
    print(f"✓ Generated {len(df)} customer records → data/customers.csv")
    print("\nSample:")
    print(df.head(5).to_string(index=False))
    print(f"\nIncome range: ₹{df['income'].min():,} — ₹{df['income'].max():,}")
    print(f"Avg past opens: {df['past_opens'].mean():.1f}")
    print(f"Cities: {df['location'].nunique()} unique")
    print(f"Product interests: {df['product_interest'].value_counts().to_dict()}")

    # Also generate dummy historical opens CSV for Foresight Oracle
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    dates = pd.date_range(end=datetime.now(), periods=180, freq="D")
    opens = np.clip(
        30 + np.linspace(0, 5, 180) + np.random.normal(0, 3, 180),
        5, 70
    ).astype(int)
    hist_df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "opens": opens})
    hist_df.to_csv("data/historical_opens.csv", index=False)
    print(f"\n✓ Generated 180-day historical opens data → data/historical_opens.csv")
