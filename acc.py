import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('prediction_results.csv')

# Display the first few rows of the dataframe to verify it's loaded correctly
print(df.head())

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(df['Difference']))
print(f"Mean Absolute Error (MAE): {mae}")

# Optional: Calculate a custom accuracy based on a threshold
accuracy_threshold = 30 # Counts within ±5 are considered accurate
accurate_predictions = np.sum(np.abs(df['Difference']) <= accuracy_threshold)
total_predictions = len(df)
accuracy_rate = (accurate_predictions / total_predictions) * 100

print(f"Accuracy (within ±{accuracy_threshold} counts): {accuracy_rate:.2f}%")
