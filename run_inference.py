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

import queue as Q
from multiprocessing import Process, Queue
import threading

import sys
import argparse
import time

import dlib
import pyautogui
import cv2

import imutils
from imutils import face_utils

import utils
from segmenter import Segmenter

from gaze_estimator import GazeEstimator

flip = True

class Screen:
    availableDisplays = {
        'Surface Pro 4': {
            'camera': [10.0, 1.0],
            'size': [20.0, 15.0],
            'coordFactors': [1.0, 1.0]
        },
        'Surface Book': {
            'camera': [14.25, -0.75],
            'size': [28.5, 19.0],
            'coordFactors': [1.0, 1.0]
        },
        'Desktop': {
            'camera': [18.0, -1.5],
            'size': [62.0, 15.0],
            'coordFactors': [1.0, 1.0]
        }
    }
    def __init__(self, display="Surface Pro 4"):
        # make sure we have a valid display
        if display is None or display not in self.availableDisplays.keys():
            err = "Bad screen provided, must be one of {}".format(
                list(self.availableDisplays.keys())
            )
            raise ValueError(err)
        # get the size of the screen (pixels)
        self.pixels = pyautogui.size()
        # Define camera and screen parameters
        self.camera = self.availableDisplays[display]['camera']
        self.size = self.availableDisplays[display]['size']
        # Conversion factors to scale centimeters to screen pixels
        self.conversion = [
            self.pixels[0] / self.size[0],
            self.pixels[1] / self.size[1]
        ]
        self.coordFactors = self.availableDisplays[display]['coordFactors']

    def cm2Px(self, coords):
        pos = [
            round(self.conversion[0]*((coords[0]+3.0) * self.coordFactors[0] + self.camera[0])),
            round(self.conversion[1]*(-coords[1] * self.coordFactors[1] + self.camera[1]))
        ]
        return (max(0,min(self.pixels[0], pos[0])),
                max(0,min(self.pixels[1], pos[1])))

def thread_func(args, detector, predictor, img_queue, result_queue):
    """Get face from image queue. This function is used for multiprocessing"""

    # Introduce mark_detector to detect landmarks.
    gaze_model = args["gaze_net"]
    eye_size = args["eye_size"]
    face_size = args["face_size"]
    inputs = args["inputs"]
    outputs = args["outputs"]
    print("[INFO] loading gaze predictor...")
    gaze_detector = GazeEstimator(
        gaze_model=gaze_model,
        eye_image_size=eye_size,
        face_image_size=face_size,
        inputs=inputs,
        outputs=outputs
    )

    # init variables
    detectorWidth = 400
    faceBoxScale = 0.15

    while True:
        # get the image
        try:
            frame = img_queue.get(timeout=1)
        except Q.Empty:
            print("Image Q empty, thread exiting!")
            return
        # update factors
        originalWidth = frame.shape[1]
        factor = originalWidth / detectorWidth
        # resize for face detection
        image = imutils.resize(frame, width=detectorWidth)
        # convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # actually run face detection
        faceboxes, scores, idx = detector.run(image, 0)
        if faceboxes is not None and len(faceboxes) > 0:
            facebox = faceboxes[0]
            confidence = scores[0]
            # get 5 landmarks
            marks = predictor(gray, facebox)
            # convert marks to np array
            marks = face_utils.shape_to_np(marks)
            leftEyeMarks = []
            rightEyeMarks = []
            # pull out left and right eye marks
            for (i, (x, y)) in enumerate(marks):
                [x,y] = [int(x*factor),int(y*factor)]
                if i == 0 or i ==1:
                    leftEyeMarks.append([x,y])
                if i == 2 or i ==3:
                    rightEyeMarks.append([x,y])

            # convert the facebox from dlib format to regular BB and
            # rescale it back to original image size
            facebox = utils.dlib_to_box(facebox, factor, faceBoxScale)
            # segment the image based on markers and facebox
            seg = Segmenter(facebox, leftEyeMarks, rightEyeMarks, frame.shape[1], frame.shape[0])
            segments = seg.getSegmentJSON()
            # detect gaze
            gaze = gaze_detector.detect_gaze(
                frame,
                segments["leftEye"],
                segments["rightEye"],
                segments["face"],
                segments["faceGrid"]
            )
            # pack result
            result = [gaze, frame]
            result_queue.put(result)
        else:
            result_queue.put(None)


def main():
    """MAIN"""
    pyautogui.FAILSAFE = False

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--num-threads", type=int, default=2,
                    help="number of threads for detecting gaze")
    ap.add_argument("-v", "--video-src", type=int, default=0,
                    help="video source index")
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

    # set up detector, predictor from dlib
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # set up multiprocessing
    num_threads = args["num_threads"]

    # Video source from webcam or video file.
    video_src = args["video_src"]
    cam = cv2.VideoCapture(video_src)
    _, sample_frame = cam.read()

    if sample_frame is None:
        print("Could not open video source", video_src)
        print("Exiting!")
        return

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    result_queue = Queue()
    tids = []

    # create the threads
    for i in range(num_threads):
        if isWindows():
            thread = threading.Thread(target=thread_func,
                                      args=(args, detector, predictor, img_queue, result_queue))
            thread.setDaemon(True)
        else:
            thread = Process(target=thread_func,
                             args=(args, detector, predictor, img_queue, result_queue))
        tids.append(thread)

    # start the threads
    for tid in tids:
        tid.start()

    # performance measurements
    numFrames = 0
    start = time.time()
    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cam.read()
        if frame_got is False:
            break

        if flip:
            # flip it so it looks like a mirror.
            frame = cv2.flip(frame, 2)

        # Feed frame to image queue.
        if img_queue.qsize() < num_threads:
            img_queue.put(frame)

        try:
            result = result_queue.get(timeout=0.05)
            if result is not None:
                # unpack result
                gaze, frame = result
                if flip:
                    gaze[0] = -gaze[0]
                #print(gaze)
                x,y = screen.cm2Px(gaze)
                #print((x,y))
                pyautogui.moveTo(x,y)
                # increment frame counter for performance measurements
                numFrames += 1
        except Q.Empty as inst:
            pass

        # Show preview. - needed for cv2.waitKey() to work!
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27: # sadly adds 1 ms of wait :(
            break

    # compute FPS and print
    end = time.time()
    diff = end - start
    print("Elapsed time:", diff)
    print("FPS:", numFrames / diff)

    # Clean up the multiprocessing process.
    for tid in tids:
        if not isWindows():
            tid.terminate()
        tid.join()

    print("All threads have finished, process exiting!")

if __name__ == '__main__':
    main()
