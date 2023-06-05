import cv2

# Open the video file
cap = cv2.VideoCapture('C:/Users/Aruay/Desktop/ra/videos_all/1.mp4')

# Get the number of frames in the video
num_frames = 0
num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print('Number of frames:', num_frames)

cap = cv2.VideoCapture('C:/Users/Aruay/Desktop/ra/videos_all/2.mp4')
num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Number of frames:', num_frames)

cap = cv2.VideoCapture('C:/Users/Aruay/Desktop/ra/videos_all/3.mp4')
num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Number of frames:', num_frames)

cap = cv2.VideoCapture('C:/Users/Aruay/Desktop/ra/videos_all/4.mp4')
num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Number of frames:', num_frames)

cap = cv2.VideoCapture('C:/Users/Aruay/Desktop/ra/videos_all/5.mp4')
num_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Number of frames:', num_frames)

# Release the video file
cap.release()

