import os
import cv2
import ast
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import re
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Conv3D, MaxPooling3D, Flatten, Input, Reshape, TimeDistributed, \
    Masking
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.interpolate import interp1d
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a dictionary that maps string labels to integer labels
label_map = {
    'go straight': 0,
    'turn left': 1,
    'turn right': 2,
    'sit': 3,
    'stand up': 4,
    'start sitting down': 5
}


def preprocess_videos(video_path, num_frames, frame_height, frame_width):
    # Get a list of all video files in the directory
    video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
    video_frames = []
    samples = []

    for video in video_files:
        new_path = os.path.join(video_path, video)
        cap = cv2.VideoCapture(new_path)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[:, :, :3]
            frame = np.expand_dims(frame, axis=0)
            video_frames.append(frame)

            if len(video_frames) == num_frames:
                sample = np.concatenate(video_frames, axis=0)
                sample = np.expand_dims(sample, axis=0)
                samples.append(sample)
                video_frames.pop(0)

        cap.release()

    return np.concatenate(samples, axis=0)


def preprocess_csv(csv_file, num_frames, video_shape):
    df = pd.read_csv(csv_file, dtype=str)

    X = df.drop(columns=['action_label', 'video_number', 'frame_number']).values
    # X = df.drop(columns=['action_label', 'video_number', 'frame_number', 'nose', 'left_eye_inner', 'left_eye_center',
    #                     'left_eye_outer', 'right_eye_inner', 'right_eye_center', 'right_eye_outer', 'left_ear',
    #                     'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
    #                     'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
    #                     'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
    #                     'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index',
    #                     'right_foot_index']).values

    # X = df.drop(columns=['action_label', 'video_number', 'frame_number', "video_number", "frame_number", "chin",
    #                     "left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip", "left_eye", "right_eye", "top_lip",
    #                     "bottom_lip", "angry", "scared", "happy", "sad", "surprised", "neutral"]).values

    Y_str = df['action_label'].values
    Y = np.array([label_map[label_str] for label_str in Y_str])

    num_samples = (X.shape[0] - num_frames + 1)
    X_samples = np.empty((num_samples, num_frames, *video_shape), dtype=X.dtype)
    Y_samples = np.empty((num_samples, num_frames), dtype=Y.dtype)

    print("pass")

    for i in range(num_samples):
        end_idx = i + num_frames
        sample_str = X[i:end_idx]
        Y_sample = Y[i:end_idx]
        sample = np.empty((num_frames,), dtype=object)

        for frame in range(sample_str.shape[0]):
            new_val = []

            for feature in range(sample_str.shape[1]):
                val = sample_str[frame, feature]
                type = ast.literal_eval(val)

                if isinstance(type, float):
                    val = float(type)
                    if val == 0.0:
                        new_val.append(-1.0)

                    else:
                        new_val.append(val)

                elif isinstance(type, list) and all(isinstance(v, float) for v in type):
                    for v in type:
                        val = float(v)
                        new_val.append(val)

                    if len(type) == 0 and feature >= 9:
                        new_val.append(-1.0)
                        new_val.append(-1.0)

                    num_points = 0
                    if feature == 0:
                        num_points = 34

                    elif feature <= 2 or feature == 4:
                        num_points = 10

                    elif feature == 3:
                        num_points = 8

                    elif feature <= 6:
                        num_points = 12

                    elif feature <= 8:
                        num_points = 24

                    for cnt in range(num_points):
                        new_val.append(-1)

                elif isinstance(type, list) and all(
                        isinstance(v, list) and all(isinstance(e, int) for e in v) for v in type):
                    for v in type:
                        for e in v:
                            new_val.append(int(e))

            sample[frame] = new_val

        # Convert list of lists to 3D numpy array
        sample = np.array([np.array(col) for col in sample], dtype=object)

        # Reshape the sample to match the expected video shape
        new_shape = (num_frames, *video_shape)
        sample_reshaped = sample.reshape(new_shape)

        X_samples[i] = sample_reshaped
        Y_samples[i] = Y_sample

    return np.array(X_samples), np.array(Y_samples)


def create_3dcnn_model(data_shape, num_classes):
    # Define two input layers
    video_input = Input(shape=data_shape, name='video_input')
    csv_input = Input(shape=data_shape, name='csv_input')

    # Define the 3D convolutional layers for the video input with dropout regularization
    conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.001))(video_input)
    drop1 = Dropout(rate=0.25)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(drop1)
    conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(0.001))(pool1)
    drop2 = Dropout(rate=0.25)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(drop2)

    # Flatten the output from the convolutional layers
    flatten_1 = Flatten()(pool2)
    flatten_2 = Flatten()(csv_input)

    # Concatenate the flattened convolutional output with the CSV input
    merged = concatenate([flatten_1, flatten_2])

    # Define the fully connected layers for the merged input with dropout regularization
    fc1 = Dense(units=512, activation='relu')(merged)
    drop4 = Dropout(rate=0.5)(fc1)
    fc2 = Dense(units=256, activation='relu')(drop4)
    drop5 = Dropout(rate=0.5)(fc2)
    output = Dense(units=num_classes, activation='softmax')(drop5)

    # Create the model with both inputs and the output
    model = Model(inputs=[video_input, csv_input], outputs=output)

    return model


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)

    # Plot loss
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='test')
    ax1.set_ylabel('Loss')

    # Determine upper bound of y-axis
    max_loss = max(history.history['loss'] + history.history['val_loss'])

    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])

    # Plot accuracy
    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='test')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])

    plt.savefig('/home/jack/Desktop/action_recognition/loss_accuracy_new.jpg', bbox_inches='tight', dpi=300)
    plt.show()


# Preprocess the videos
num_frames = 8
frame_height = 6
frame_width = 12

video_data = preprocess_videos('/home/jack/Desktop/action_recognition/videos_all', num_frames, frame_height,
                               frame_width)
test_video_data = preprocess_videos('/home/jack/Desktop/action_recognition/videos_all/test', num_frames, frame_height,
                                    frame_width)
print("videos done")
print(video_data.shape)
print(test_video_data.shape)

csv_data, labels = preprocess_csv('/home/jack/Desktop/action_recognition/train_train_modified.csv', num_frames, video_data.shape[-3:])
test_csv_data, test_labels = preprocess_csv('/home/jack/Desktop/action_recognition/train_test.csv', num_frames, video_data.shape[-3:])
print("csv done")
print(csv_data.shape)
print(labels.shape)

num_classes = len(label_map)

# Split the data into training and validation sets
train_labels, val_labels, train_video_data, val_video_data, train_csv_data, val_csv_data = train_test_split(
    labels,
    video_data,
    csv_data,
    test_size=0.2
)

# Create the model
model = create_3dcnn_model(train_video_data.shape[-4:], num_classes)

# Compile and train the model
batch_size = 8
epochs = 10
# opt = SGD(lr=0.0001)
opt = Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print('train_video shape:', train_video_data.shape)
print('train_csv shape:', train_csv_data.shape)
print('train_labels shape:', train_labels.shape)

train_labels = to_categorical(train_labels, num_classes)
train_labels = train_labels.reshape((train_video_data.shape[0], num_frames, num_classes))
val_labels = to_categorical(val_labels, num_classes)
val_labels = val_labels.reshape((val_video_data.shape[0], num_frames, num_classes))
test_labels = to_categorical(test_labels, num_classes)
test_labels = test_labels.reshape((test_video_data.shape[0], num_frames, num_classes))

aggregated_predictions = np.mean(train_labels, axis=1)
rounded = np.round(aggregated_predictions)
train_labels = rounded.astype(int)

aggregated_predictions = np.mean(val_labels, axis=1)
rounded = np.round(aggregated_predictions)
val_labels = rounded.astype(int)

aggregated_predictions = np.mean(test_labels, axis=1)
rounded = np.round(aggregated_predictions)
test_labels = rounded.astype(int)

print(val_labels)
print(val_labels.shape)
print(model.input_shape)

train_video_data = 2 * (train_video_data / 255.0) - 1
val_video_data = 2 * (val_video_data / 255.0) - 1
test_video_data = 2 * (test_video_data / 255.0) - 1

train_video_data_tensor = tf.convert_to_tensor(train_video_data)
val_video_data_tensor = tf.convert_to_tensor(val_video_data)
test_video_data_tensor = tf.convert_to_tensor(test_video_data)

train_csv_data = train_csv_data.tolist()
val_csv_data = val_csv_data.tolist()
test_csv_data = test_csv_data.tolist()

train_csv_data_tensor = tf.convert_to_tensor(train_csv_data)
train_labels_tensor = tf.convert_to_tensor(train_labels)

val_csv_data_tensor = tf.convert_to_tensor(val_csv_data)
val_labels_tensor = tf.convert_to_tensor(val_labels)

test_csv_data_tensor = tf.convert_to_tensor(test_csv_data)
test_labels_tensor = tf.convert_to_tensor(test_labels)

history = model.fit(x=[train_video_data_tensor, train_csv_data_tensor], y=train_labels_tensor,
                    validation_data=([val_video_data_tensor, val_csv_data_tensor], val_labels_tensor),
                    batch_size=batch_size, epochs=epochs)

model.save('/home/jack/Desktop/action_recognition/weights_new.h5')

plot_history(history)
model.evaluate([test_video_data_tensor, test_csv_data_tensor], test_labels_tensor, return_dict=True)

# Get the predicted labels from the model on the test data
y_pred = model.predict([test_video_data_tensor, test_csv_data_tensor])
y_pred = np.argmax(y_pred, axis=1)

# Get the true labels for the test data
y_true = np.argmax(test_labels_tensor, axis=1)

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Visualize the confusion matrix as an image
plt.matshow(cm, cmap=plt.cm.Blues)

# Add numeric values to each cell in the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center", color="black")

# Add labels to the matrix
plt.title('Confusion matrix')
plt.colorbar()

classes = ['go straight', 'turn left', 'turn right', 'sit', 'stand up', 'start sitting down']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.ylabel('True label')
plt.xlabel('Predicted label')

# Save the image file
plt.savefig('/home/jack/Desktop/action_recognition/confusion_matrix_2.jpg')
plt.show()
