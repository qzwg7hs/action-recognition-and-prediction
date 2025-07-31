import cv2
import csv

cap = cv2.VideoCapture("C:/Users/Aruay/Desktop/ra/videos/obstacles/5.mp4")

# Open a CSV file for writing and write header
csv_file = open("C:/Users/Aruay/Desktop/ra/videos/obstacles/action5.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
header = ["action_label"]
csv_writer.writerow(header)

frame_number = 0
delay = 25

while True:
    ret, frame = cap.read()
    if not ret:
        break

    action_val = []
    print(frame_number)

    if frame_number <= 65:
        action_val.append("go_straight")

    elif frame_number <= 105:
        action_val.append("turn_left")

    elif frame_number <= 130:
        action_val.append("turn_right")

    elif frame_number <= 160:
        action_val.append("go_straight")

    elif frame_number <= 175:
        action_val.append("turn_left")

    elif frame_number <= 218:
        action_val.append("go_straight")

    csv_writer.writerow(action_val)

    frame_number += 1
    frame = cv2.resize(frame, (700, 400))
    cv2.imshow('Video', frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
