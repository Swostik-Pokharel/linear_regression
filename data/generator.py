import numpy as np
import pandas as pd

# Get user input
num_variables = int(input("How many variables do you want? "))
num_datapoints = int(input("How many datapoints do you want? "))
filename = input("What should be the filename? (don't add .csv): ")

# Set random seed for reproducibility
np.random.seed(42)

# Generate variable names (x, y, z, ...)
variable_names = [chr(120 + i) for i in range(num_variables - 1)]  # x, y, z, ...
target_name = chr(120 + num_variables - 1)  # last variable is the target

# Generate random data for independent variables
data_dict = {}
for var in variable_names:
    data_dict[var] = np.random.uniform(0, 100, num_datapoints)

# Generate target variable with linear relationship plus noise
# target = sum of all variables + noise
target = sum(data_dict.values()) + np.random.normal(0, 15, num_datapoints)
data_dict[target_name] = target

# Create DataFrame
data = pd.DataFrame(data_dict)

# Save to CSV
data.to_csv(f"{filename}.csv", index=False)

print(f"\nCSV file '{filename}.csv' created successfully!")
print(f"Generated {len(data)} data points with {num_variables} variables")
print(f"\nFirst few rows:")
print(data.head())
