# Human Action Recognition & Prediction with 3D CNN

This project uses a dual-input 3D Convolutional Neural Network (3D CNN) to recognize and predict human actions from video clips and extracted body/emotion features. It fuses temporal video data with spatial landmarks and emotion vectors to improve classification accuracy.

## Dataset

- 5 action classes: *looking straight, turning left, turning right, sitting, sitting/standing transition*
- Each sample = 16 frames (30 FPS → 0.53s)
- Extracted features:
  - Facial landmarks (dlib)
  - Body joints (MediaPipe)
  - Emotion vectors (VGG face model)

## Model Architecture

- 2-stream input: raw video (3DCNN) + CSV features (emotion, pose)
- Video processed by 3D CNN blocks with 3×3×3 filters
- CSV features merged after flattening
- 2 Dense layers, softmax classification

## Results

- Testing accuracy: **77.03%**
- Confusion matrix and model architecture in `visualizations/`
