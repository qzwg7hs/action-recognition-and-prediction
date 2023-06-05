from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import pandas as pd

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier("C:/Users/Aruay/Desktop/ra/project/haarcascade_frontalface_default.xml")
model = load_model("C:/Users/Aruay/Desktop/ra/project/pretrainedVGGnet.hdf5")
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

camera = cv2.VideoCapture("C:/Users/Aruay/Desktop/ra/videos/obstacles/face_videos/1.mp4")

predictions = []

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	if not grabbed:
		break

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# initialize the canvas for the visualization, then clone
	# the frame so we can draw on it
	canvas = np.zeros((220, 300, 3), dtype="uint8")
	frameClone = frame.copy()

	# detect faces in the input frame, then clone the frame so that
	# we can draw on it
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	temp = []

	# ensure at least one face was found before continuing
	if len(rects) > 0:
		# determine the largest face area
		rect = sorted(rects, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
		(fX, fY, fW, fH) = rect

		# extract the face ROI from the image, then pre-process
		# it for the network
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (48, 48))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

		# make a prediction on the ROI, then lookup the class label
		preds = model.predict(roi)[0]
		label = EMOTIONS[preds.argmax()]

		# loop over the labels + probabilities and draw them
		for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
			# construct the label text
			text = "{}: {:.2f}%".format(emotion, prob * 100)

			# draw the label + probability bar on the canvas
			w = int(prob * 300)
			cv2.rectangle(canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
			cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

			temp.append(prob * 100)

		# draw the label on the frame
		cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

	if len(temp) == 0:
		for i in range(6):
			temp.append(0)

	predictions.append(temp)

	# show our classifications + probabilities
	cv2.imshow("Face", frameClone)
	cv2.imshow("Probabilities", canvas)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# Create dataframe
data = pd.DataFrame(predictions, columns=EMOTIONS)

# Save to CSV
data.to_csv("C:/Users/Aruay/Desktop/ra/videos/obstacles/emotions1.csv", index=False)

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
