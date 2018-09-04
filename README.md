# Gaze Estimation

This repo shows how to find faces and landmarks and then use them to
predict gaze.

## Dependence
- TensorFlow 1.4
- DLIB
- OpenCV 3.3
- Python 3

The code is tested under Ubuntu 16.04.

## How it works

There are three major steps:

1. Face detection. A face detector is adopted to provide a face box
   containing a human face. Then the face box is expanded and
   transformed to a square to suit the needs of later steps.

2. Facial landmark detection. A custom trained facial landmark
   detector based on TensorFlow is responsible for output 68 facial
   landmarks.

3. Pose estimation. Once we got the 68 facial landmarks, a mutual PnP
   algorithms is adopted to calculate the pose.

## Miscellaneous
- The marks is detected frame by frame, which result in small variance
  between adjacent frames. This makes the pose unstaible. A Kalman
  filter is used to solve this problem, you can draw the original pose
  to observe the difference.

## How To Use

```bash
usage: run_inference.py [-h] [-m] [-s] [-d] [-g GAZE_NET] [-e EYE_SIZE]
                        [-f FACE_SIZE] [-i INPUTS] [-o OUTPUTS] -p
                        SHAPE_PREDICTOR

optional arguments:
  -h, --help            show this help message and exit
  -m, --draw-markers    draw the 5 face landmarks
  -s, --draw-segmented  draw the eye and face bounding boxes
  -d, --detect-gaze     enable gaze detection
  -g GAZE_NET, --gaze-net GAZE_NET
                        path to frozen gaze predictor model
  -e EYE_SIZE, --eye-size EYE_SIZE
                        input image sizes for the eyes
  -f FACE_SIZE, --face-size FACE_SIZE
                        input image size for the face
  -i INPUTS, --inputs INPUTS
                        input tensor names, comma separated
  -o OUTPUTS, --outputs OUTPUTS
                        output tensor names, comma separated
  -p SHAPE_PREDICTOR, --shape-predictor SHAPE_PREDICTOR
                        path to facial landmark predictor
```

We have the shape predictor stored in
[./model/shape_predictor_5_face_landmarks.dat](./model/shape_predictor_5_face_landmarks.dat),
which you must provide as the shape predictor argument.

Press `ESCAPE` to exit the progam - at exit it will print the FPS it
achieved.
