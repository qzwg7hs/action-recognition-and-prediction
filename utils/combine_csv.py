import pandas as pd

df1 = pd.read_csv("C:/Users/Aruay/Desktop/ra/videos/obstacles/train.csv")
df2 = pd.read_csv("C:/Users/Aruay/Desktop/ra/videos/obstacles/video5.csv")

# Combine the two dataframes
df_combined = pd.concat([df1, df2], ignore_index=True)

# Write the combined dataframe to a new CSV file
df_combined.to_csv("C:/Users/Aruay/Desktop/ra/videos/obstacles/train.csv", index=False)
