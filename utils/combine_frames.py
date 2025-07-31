import cv2
import os

# Get all the frame files
frame_files = []
for f in os.listdir("C:/Users/Aruay/Desktop/ra/videos/obstacles/face_frames/"):
    if f.endswith('jpg'):
        frame_files.append(f)


def frame_sorter(frame_filename):
    filename, file_extension = os.path.splitext(frame_filename)
    frame_number = int(filename.split('frame')[-1])
    return frame_number


frame_files.sort(key=frame_sorter)

# Define the codec and create VideoWriter object
frame_width = 64  # Define the width of the frames
frame_height = 64  # Define the height of the frames

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("C:/Users/Aruay/Desktop/ra/videos/obstacles/face_videos/1.mp4",
                      fourcc, 30, (frame_width, frame_height))

# Loop through frames and write to video
for file in frame_files:
    full = os.path.join("C:/Users/Aruay/Desktop/ra/videos/obstacles/face_frames", file)
    frame = cv2.imread(full)
    if frame is not None:
        out.write(frame)
        cv2.imshow('video', frame)

# Release everything
out.release()
cv2.destroyAllWindows()
