import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Flatten, Input, Reshape, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Define a dictionary that maps string labels to integer labels
label_map = {
    'go_straight': 0,
    'turn_left': 1,
    'turn_right': 2
}


def preprocess_videos(video_path, num_frames, frame_height, frame_width):
    # Get a list of all video files in the directory
    video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]

    video_data = []
    video_frames = []
    total_frames = 0
    frame_num = 0

    for video in video_files:
        new_path = os.path.join(video_path, video)

        # print("video = ", new_path)

        cap = cv2.VideoCapture(new_path)
        total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            resized_frame = cv2.resize(frame, (frame_width, frame_height))
            video_frames.append(resized_frame)
            frame_num += 1

            # initial version (inputs all frames sequentially, not in chunks)
            # video_data.append(video_frames)

            # updated version (inputs in chunks of 16 frames)
            if frame_num >= num_frames:
                video_data.append(video_frames[-num_frames:])
                frame_num -= 1

        cap.release()

    return np.array(video_data), total_frames


def preprocess_csv(csv_file, num_frames):
    data = pd.read_csv(csv_file)
    labels = []
    frame_num = 0
    csv_labels = []
    csv_label_row = []

    # Get all columns from the CSV
    columns = data.columns

    for index, row in data.iterrows():
        action_label = row['action_label']
        int_label = label_map[action_label]
        frame_num += 1

        # Get values for all columns
        row_values = []
        for col in columns:
            pattern = r"\[\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]](?:,\[[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)])*"

            for rows in data[col]:

                # Check if rows is a string before attempting regex
                if isinstance(rows, str):
                    if re.match(pattern, rows):
                        # Find all matches and convert to floats
                        pairs = re.findall(pattern, rows)
                        coord_pairs = [[float(x), float(y)] for x, y in pairs]
                        row_values.append(coord_pairs)
                        # for x, y in re.findall(pattern, rows):
                        #    row_values.append([float(x), float(y)])

                    else:
                        if rows in label_map:
                            row_values.append(label_map[rows])
                        else:
                            pass

                else:
                    num = float(rows)
                    row_values.append(num)

            # row_values.append(row[col])

        csv_labels.append(int_label)
        if frame_num >= num_frames:
            labels.append(csv_labels[-num_frames:])
            csv_label_row.append(row_values[-num_frames:])
            frame_num -= 1

    return np.array(labels), np.array(csv_label_row, dtype=object)


# Preprocess the videos
num_frames = 16
frame_height = 54
frame_width = 96

video_data, total_frames = preprocess_videos('C:/Users/Aruay/Desktop/ra/videos_all', num_frames, frame_height,
                                             frame_width)
print("videos done")
labels, csv_data = preprocess_csv('C:/Users/Aruay/Desktop/ra/videos/obstacles/train.csv', num_frames)
print("csv done")

# Print the shapes of the input data and labels
print('Input data shape:', video_data.shape)
print('Labels shape:', labels.shape)
print('CSV shape:', csv_data.shape)

# Split the data into training and validation sets
# train_data, val_data, train_labels, val_labels = train_test_split(video_data, labels, test_size=0.2)
train_labels, val_labels, train_video_data, val_video_data, train_csv_data, val_csv_data = train_test_split(
    labels,
    video_data,
    csv_data,
    test_size=0.2
)

# One-hot encode the labels
num_classes = len(label_map)
# val_video_data = to_categorical(val_video_data, num_classes)
val_labels = to_categorical(val_labels, num_classes)

# Create the model
input_shape = (num_frames, frame_height, frame_width, 3)

# Convert inputs to Tensors
val_video_data = tf.convert_to_tensor(val_video_data)
val_labels = tf.convert_to_tensor(val_labels)

print('test_video shape:', val_video_data.shape)
print('test_csv shape:', val_csv_data.shape)
print('test_labels shape:', val_labels.shape)

# Load the model
model = load_model('C:/Users/Aruay/Desktop/ra/project/weights.h5')

# Evaluate the model
loss, accuracy = model.evaluate(val_video_data, val_labels)
print('Test accuracy:', accuracy)


