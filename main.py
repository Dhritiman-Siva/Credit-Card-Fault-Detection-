import pandas as pd

# Load the dataset directly from the zip file
df = pd.read_csv('creditcard.csv.zip')

# Display basic information
print(df.head())
print("\nShape:", df.shape)
