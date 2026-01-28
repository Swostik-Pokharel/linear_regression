import numpy as np
import pandas as pd

# Get user input
num_variables = int(input("How many variables do you want? "))
num_datapoints = int(input("How many datapoints do you want? "))
scatter_level = int(input("Choose scatter level (1-5, where 5 is most scattered): "))
filename = input("What should be the filename? (don't add .csv): ")

# Validate scatter level
scatter_level = max(1, min(5, scatter_level))

# Set random seed for reproducibility
np.random.seed(42)

# Generate variable names (x, y, z, ...)
variable_names = [chr(120 + i) for i in range(num_variables - 1)]  # x, y, z, ...
target_name = chr(120 + num_variables - 1)  # last variable is the target

# Generate random data for independent variables
data_dict = {}
for var in variable_names:
    data_dict[var] = np.random.uniform(0, 100, num_datapoints)

# Define noise levels based on scatter setting
# Level 1: Very tight correlation (low noise)
# Level 5: Very scattered (high noise)
noise_multipliers = {
    1: 5,  # Very tight
    2: 20,  # Moderate
    3: 40,  # Scattered
    4: 70,  # Very scattered
    5: 120,  # Extremely scattered
}

noise_std = noise_multipliers[scatter_level]

# Generate target variable with linear relationship plus noise
# target = sum of all variables + noise
target = sum(data_dict.values()) + np.random.normal(0, noise_std, num_datapoints)
data_dict[target_name] = target

# Create DataFrame
data = pd.DataFrame(data_dict)

# Save to CSV
data.to_csv(f"{filename}.csv", index=False)

print(f"\nCSV file '{filename}.csv' created successfully!")
print(f"Generated {len(data)} data points with {num_variables} variables")
print(f"Scatter level: {scatter_level} (noise std: {noise_std})")
print(f"\nFirst few rows:")
print(data.head())

# Optional: Show correlation
print(
    f"\nCorrelation between {variable_names[0]} and {target_name}: {data[variable_names[0]].corr(data[target_name]):.3f}"
)
