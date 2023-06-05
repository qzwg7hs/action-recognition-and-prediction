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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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


def create_3dcnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Reshape((16, -1)))  # Add reshape layer to create a new time dimension
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

    return model


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
train_csv_data = to_categorical(train_csv_data, num_classes)
val_csv_data = to_categorical(val_csv_data, num_classes)
val_labels = to_categorical(val_labels, num_classes)

# Create the model
input_shape = (num_frames, frame_height, frame_width, 3)
model = create_3dcnn_model(input_shape, num_classes)

# Convert inputs to Tensors
train_video_data = tf.convert_to_tensor(train_video_data)
val_video_data = tf.convert_to_tensor(val_video_data)
val_labels = tf.convert_to_tensor(val_labels)
train_csv_data = train_csv_data.astype(np.float32)
train_csv_data = tf.convert_to_tensor(train_csv_data)
val_csv_data

# Compile and train the model
batch_size = 16
epochs = 50
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('train_video shape:', train_video_data.shape)
print('train_csv shape:', train_csv_data.shape)
print('train_labels shape:', train_labels.shape)

history = model.fit(train_video_data, train_csv_data, validation_data=(val_video_data, val_csv_data),
                    batch_size=batch_size, epochs=epochs)

model.save('C:/Users/Aruay/Desktop/ra/project/weights.h5')

# Accuracy vs Epoch plot
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Confusion matrix
y_pred = model.predict(val_video_data)
cm = confusion_matrix(val_labels, y_pred)

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(set(val_labels)))
plt.xticks(tick_marks, set(val_labels))
plt.yticks(tick_marks, set(val_labels))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Classification report

y_pred = model.predict(val_video_data)
report = classification_report(val_labels, y_pred)
print(report)
