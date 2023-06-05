import face_recognition
import cv2
import csv

cap = cv2.VideoCapture("C:/Users/Aruay/Desktop/ra/videos/obstacles/1.mp4")

# Open a CSV file for writing and write header
csv_file = open("C:/Users/Aruay/Desktop/ra/videos/obstacles/facial_points1.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
header = ["video_number", "frame_number", "chin", "left_eyebrow", "right_eyebrow", "nose_bridge",
          "nose_tip", "left_eye", "right_eye", "top_lip", "bottom_lip"]
csv_writer.writerow(header)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("C:/Users/Aruay/Desktop/ra/videos/obstacles/face_videos/1.mp4",
                      fourcc, 30, (64, 64))

black = cv2.imread("C:/Users/Aruay/Desktop/ra/project/black.jpg")

frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_landmarks_list = face_recognition.face_landmarks(frame)
    face_locations = face_recognition.face_locations(frame)

    row_data = [1, frame_number]

    if face_landmarks_list:
        for feature in header[2:]:
            points = []
            for point in face_landmarks_list[0][feature]:
                pair = [point[0], point[1]]
                points.append(pair)
            row_data.append(points)
    else:
        # If no face landmarks detected, write an empty row
        row_data.extend([[]] * (len(header) - 2))

    csv_writer.writerow(row_data)

    frame_number += 1

    cv2.imshow('Video', frame)

    if face_locations:
        for face in face_locations:
            print(face)
            x1 = face[0]
            y1 = face[3]
            x2 = face[2]
            y2 = face[1]

            frame = frame[(x1-15):(x2+15), (y1-15):(y2+15)]

            # change frame size to same value
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            out.write(frame)

        cv2.imshow('Face', frame)

    else:
        black = cv2.resize(black, (64, 64))
        out.write(black)
        cv2.imshow('Face', black)

    if cv2.waitKey(25) == 13:
        break


cap.release()
cv2.destroyAllWindows()
csv_file.close()
