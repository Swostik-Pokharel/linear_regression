import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate random x values
x = np.random.uniform(0, 100, 100)

# Generate y values with a linear relationship plus some noise
# y = 2x + 10 + noise
y = 2 * x + 10 + np.random.normal(0, 15, 100)

# Create DataFrame
data = pd.DataFrame({"x": x, "y": y})

# Save to CSV
data.to_csv("sample_data.csv", index=False)

print("CSV file 'sample_data.csv' created successfully!")
print(f"Generated {len(data)} data points")
print(f"\nFirst few rows:")
print(data.head())
