"""
STEP 1: Generate fake (but realistic) retail sales data.
This creates a CSV file you fully control - no external download needed.
Run: python data_generator.py
"""

import random
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)

regions = ["North", "South", "East", "West", "Central"]
categories = ["Shampoo", "Detergent", "Toothpaste", "Soap", "Snacks", "Beverages"]
products = {
    "Shampoo": ["Clinic Plus", "Head & Shoulders", "Sunsilk"],
    "Detergent": ["Surf Excel", "Ariel", "Tide"],
    "Toothpaste": ["Colgate", "Pepsodent", "Closeup"],
    "Soap": ["Lifebuoy", "Dove", "Lux"],
    "Snacks": ["Lays", "Kurkure", "Bingo"],
    "Beverages": ["Coca-Cola", "Pepsi", "Sprite"],
}

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 6, 30)

rows = []
current = start_date
while current <= end_date:
    for region in regions:
        for category in categories:
            product = random.choice(products[category])
            base_units = random.randint(50, 500)

            # Intentionally inject a big sales DROP here so your agent
            # has something interesting to "detect" later.
            if region == "South" and category == "Detergent" and current.year == 2025 and current.month == 5:
                base_units = int(base_units * 0.4)

            price = round(random.uniform(30, 300), 2)
            revenue = round(base_units * price, 2)

            rows.append(
                {
                    "date": current.strftime("%Y-%m-%d"),
                    "region": region,
                    "category": category,
                    "product": product,
                    "units_sold": base_units,
                    "price": price,
                    "revenue": revenue,
                }
            )
    current += timedelta(days=7)  # weekly snapshots

df = pd.DataFrame(rows)
df.to_csv("retail_sales_raw.csv", index=False)
print(f"Generated {len(df)} rows -> retail_sales_raw.csv")
print(df.head())