import pandas as pd

# Read the CSV into a DataFrame
df = pd.read_csv("C:/Users/Aruay/Desktop/ra/videos/obstacles/pose_data1.csv")

# Check the current column names
print(df.columns)

# Rename the columns
new_columns = ['nose','left_eye_inner','left_eye','left_eye_outer','right_eye_inner','right_eye','right_eye_outer','left_ear','right_ear','mouth_left','mouth_right','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_pinky','right_pinky','left_index','right_index','left_thumb','right_thumb','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_heel','right_heel','left_foot_index','right_foot_index']
df.columns = new_columns

# Check the new column names
print(df.columns)

# Save the DataFrame to a new CSV
df.to_csv("C:/Users/Aruay/Desktop/ra/videos/obstacles/pose_data1_new.csv", index=False)
