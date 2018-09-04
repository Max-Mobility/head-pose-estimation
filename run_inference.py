#!/usr/bin/env python
"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from os_detector import detect_os, isWindows
# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

from multiprocessing import Process, Queue
import threading

from imutils import face_utils
from imutils.video import VideoStream
import dlib
import argparse
import imutils
import time
import numpy as np

import pyautogui
import cv2

import utils
from gaze_estimator import GazeEstimator
from segmenter import Segmenter

class Screen:
    availableDisplays = {
        'Surface Pro 4': {
            'camera': [13.0, -0.7],
            'size': [26.0, 17.5],
            'coordFactor': 1.0
        },
        'Surface Book': {
            'camera': [14.25, -0.75],
            'size': [28.5, 19.0],
            'coordFactor': 1.0
        },
        'Desktop': {
            'camera': [18.0, -1.5],
            'size': [62.0, 15.0],
            'coordFactor': 1.0
        }
    }
    def __init__(self, display="Surface Pro 4"):
        # make sure we have a valid display
        if display is None or display not in self.availableDisplays.keys():
            raise ValueError("Bad screen provided, must be one of {}".format(list(self.availableDisplays.keys())))
        # get the size of the screen (pixels)
        self.screenSize = pyautogui.size()
        # Define camera and screen parameters
        self.xCameraOffsetCm = self.availableDisplays[display]['camera'][0]
        self.yCameraOffsetCm = self.availableDisplays[display]['camera'][1]
        self.wScreenCm = self.availableDisplays[display]['size'][0]
        self.hScreenCm = self.availableDisplays[display]['size'][1]
        # Conversion factors to scale centimeters to screen pixels
        self.xCm2Px = self.screenSize[0]/self.wScreenCm
        self.yCm2Px = self.screenSize[1]/self.hScreenCm
        self.coordFactor = self.availableDisplays[display]['coordFactor']

    def cm2Px(self, coords):
        pos = [
            round(self.xCm2Px*(coords[0] * self.coordFactor + self.xCameraOffsetCm)),
            round(self.yCm2Px*(-1*coords[1] * self.coordFactor + self.yCameraOffsetCm))
        ]
        return (max(0,min(self.screenSize[0], pos[0])),
                max(0,min(self.screenSize[1], pos[1])))

def get_face(detector, img_queue, result_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        result = detector.run(image, 0)
        result_queue.put(result)


def main():
    """MAIN"""
    pyautogui.FAILSAFE = False

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video-src", type=int, default=0,
                    help="video source index")
    ap.add_argument("-c", "--draw-confidence", action="store_true", default=False,
                    help="draw the confidence on the face")
    ap.add_argument("-m", "--draw-markers", action="store_true", default=False,
                    help="draw the 5 face landmarks")
    ap.add_argument("-s", "--draw-segmented", action="store_true", default=False,
                    help="draw the eye and face bounding boxes")
    ap.add_argument("-d", "--detect-gaze", action="store_true", default=False,
                    help="enable gaze detection")
    ap.add_argument("-g", "--gaze-net", type=str, default='model/mobileNet.pb',
                    help="path to frozen gaze predictor model")
    ap.add_argument("-r", "--screen", type=str, default='Surface Pro 4',
                    help="screen that we are testing on")
    ap.add_argument("-e", "--eye-size", type=int, default=224,
                    help="input image sizes for the eyes")
    ap.add_argument("-f", "--face-size", type=int, default=224,
                    help="input image size for the face")
    ap.add_argument("-i", "--inputs", type=str, default='input_1,input_2,input_3,input_4',
                    help="input tensor names, comma separated")
    ap.add_argument("-o", "--outputs", type=str, default='output_node00',
                    help="output tensor names, comma separated")
    ap.add_argument("-p", "--shape-predictor", type=str,
                    default='./model/shape_predictor_5_face_landmarks.dat',
                    help="path to facial landmark predictor")
    args = vars(ap.parse_args())

    screen = Screen(args["screen"])

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])


    # Video source from webcam or video file.
    video_src = args["video_src"]
    cam = cv2.VideoCapture(video_src)
    _, sample_frame = cam.read()

    if sample_frame is None:
        print("Could not open video source", video_src)
        print("Exiting!")
        return

    # Introduce mark_detector to detect landmarks.
    gaze_model = args["gaze_net"]
    eye_size = args["eye_size"]
    face_size = args["face_size"]
    inputs = args["inputs"]
    outputs = args["outputs"]
    gaze_detector = GazeEstimator(
        gaze_model=gaze_model,
        eye_image_size=eye_size,
        face_image_size=face_size,
        inputs=inputs,
        outputs=outputs
    )

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    result_queue = Queue()
    #img_queue.put(sample_frame)

    if isWindows():
        thread = threading.Thread(target=get_face, args=(detector, img_queue, result_queue))
        thread.daemon = True
        thread.start()
    else:
        box_process = Process(target=get_face,
                              args=(detector, img_queue, result_queue))
        box_process.start()

    detectorWidth = 400
    originalWidth = sample_frame.shape[1]
    factor = originalWidth / detectorWidth
    faceBoxScale = 0.15
    # performance measurements
    numFrames = 0
    start = time.time()
    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cam.read()
        if frame_got is False:
            break

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Feed frame to image queue.
        image = imutils.resize(frame, width=detectorWidth)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_queue.put(gray)

        # Get face from box queue.
        results = result_queue.get()
        # unpack results
        boxes, scores, idx = results

        if boxes is not None and len(boxes) > 0:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            facebox = boxes[0]
            confidence = scores[0]
            shape = predictor(gray, facebox)
            shape = face_utils.shape_to_np(shape)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
            leftEyeMarks = []
            rightEyeMarks = []
            for (i, (x, y)) in enumerate(shape):
                [x,y] = [int(x*factor),int(y*factor)]
                if i == 0 or i ==1:
                    leftEyeMarks.append([x,y])
                if i == 2 or i ==3:
                    rightEyeMarks.append([x,y])
                if args["draw_markers"]:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # convert the facebox from dlib format to regular BB and
            # rescale it back to original image size
            facebox = utils.dlib_to_box(facebox, factor, faceBoxScale)
            #draw the confidence
            if args["draw_confidence"]:
                [x,y,_,_] = facebox
                cv2.putText(frame, str(confidence), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            # segment the image based on markers and facebox
            seg = Segmenter(facebox, leftEyeMarks, rightEyeMarks, frame.shape[1], frame.shape[0])
            segments = seg.getSegmentJSON()
            if segments is not None and args["draw_segmented"]:
                utils.draw_box(frame, segments["leftEye"])
                utils.draw_box(frame, segments["rightEye"])
                utils.draw_box(frame, segments["face"])

            # detect gaze
            if segments is not None and args["detect_gaze"]:
                gaze = gaze_detector.detect_gaze(
                    frame,
                    segments["leftEye"],
                    segments["rightEye"],
                    segments["face"],
                    segments["faceGrid"]
                )
                gaze[0] = -gaze[0]
                #print(gaze)
                x,y = screen.cm2Px(gaze)
                #print((x,y))
                pyautogui.moveTo(x,y)
        # increment frame counter for performance measurements
        numFrames += 1
        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27: # sadly adds 1 ms of wait :(
            break

    end = time.time()
    diff = end - start
    print("Elapsed time:", diff)
    print("FPS:",numFrames/diff)

    # Clean up the multiprocessing process.
    if not isWindows():
        box_process.terminate()
        box_process.join()


if __name__ == '__main__':
    main()
