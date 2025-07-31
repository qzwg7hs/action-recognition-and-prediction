import face_recognition
import cv2
import os

cap = cv2.VideoCapture("C:/Users/Aruay/Desktop/ra/videos/obstacles/1.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("C:/Users/Aruay/Desktop/ra/videos/obstacles/face_videos/1.mp4",
                      fourcc, 30, (64, 64))

black = cv2.imread("C:/Users/Aruay/Desktop/ra/project/black.jpg")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_locations = face_recognition.face_locations(frame)

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

        cv2.imshow('Video', frame)

    else:
        black = cv2.resize(black, (64, 64))
        out.write(black)
        cv2.imshow('Video', black)

    if cv2.waitKey(25) == 13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
