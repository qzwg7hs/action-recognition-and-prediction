import csv
import cv2

cap = cv2.VideoCapture("C:/Users/Aruay/Desktop/ra/videos/obstacles/5.mp4")
csv_file = open("C:/Users/Aruay/Desktop/ra/videos/obstacles/emotions5_next.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

#header = ["video_number", "frame_number", "chin", "left_eyebrow", "right_eyebrow", "nose_bridge",
#          "nose_tip", "left_eye", "right_eye", "top_lip", "bottom_lip"]

header = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

csv_writer.writerow(header)
frame_number = 220

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)
for i in range(frame_number, total_frames):
    row_data = []
    row_data.extend([0] * (len(header)))
    csv_writer.writerow(row_data)

cap.release()
cv2.destroyAllWindows()
csv_file.close()
