import pandas as pd

# Read Dataset A
df1 = pd.read_csv('C:/Users/Aruay/Desktop/ra/videos/obstacles/video5.csv')

# Read Dataset B
df2 = pd.read_csv('C:/Users/Aruay/Desktop/ra/videos/obstacles/action5.csv')

# Get the columns from each dataframe
cols1 = list(df1.columns)
cols2 = list(df2.columns)

# Combine the columns in order
cols = cols1 + cols2

# Join the dataframes
df = df1.join(df2, how='inner')

# Reorder the columns
df = df[cols]

# Save the combined dataset
df.to_csv('C:/Users/Aruay/Desktop/ra/videos/obstacles/video5.csv', index=False)
