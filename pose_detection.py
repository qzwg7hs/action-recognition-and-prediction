import cv2
import mediapipe as mp
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create list to store landmark data
landmarks = []

landmark_names = ['nose', 'left_eye_inner', 'left_eye_center', 'left_eye_outer', 'right_eye_inner', 'right_eye_center',
                  'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
                  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky',
                  'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
                  'left_foot_index', 'right_foot_index']

cap = cv2.VideoCapture("C:/Users/Aruay/Desktop/ra/videos/obstacles/5.mp4")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
pTime = 0

frame_count = 0

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            # continue
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        points = []

        # Get landmark coordinates
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                points.append([landmark.x, landmark.y])

        if len(points) == 0:
            points.extend([[]] * (len(landmark_names)))

        landmarks.append(points)
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(25) == 13:
            break

        frame_count += 1

# Create pandas DataFrame
df = pd.DataFrame(landmarks, columns=landmark_names)

# Save to CSV
df.to_csv("C:/Users/Aruay/Desktop/ra/videos/obstacles/pose_data5.csv", index=False)

cap.release()
cv2.destroyAllWindows()
