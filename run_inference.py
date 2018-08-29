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

import argparse
import numpy as np

import pyautogui

import cv2
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
from gaze_estimator import GazeEstimator

from segmenter import Segmenter

CNN_INPUT_SIZE = 128

class Screen:
    def __init__(self, screenSize = (3000,2000)):
        self.screenSize = screenSize
        #Define camera and screen parameters
        # for Surface Book
        self.xCameraOffsetCm = 14.25
        self.yCameraOffsetCm = -0.75
        self.wScreenCm = 28.5
        self.hScreenCm = 19.0
        # for william desktop
        self.xCameraOffsetCm = 18.0
        self.yCameraOffsetCm = -1.5
        self.wScreenCm = 25.0
        self.hScreenCm = 15.0
        #Conversion factors to scale centimeters to screen pixels
        self.xCm2Px = self.screenSize[0]/self.wScreenCm
        self.yCm2Px = self.screenSize[1]/self.hScreenCm

    def cm2Px(self, coords):
        pos = [
            round(self.xCm2Px*(coords[0] + self.xCameraOffsetCm)),
            round(self.yCm2Px*(-1*coords[1] + self.yCameraOffsetCm))
        ]
        return (max(0,min(self.screenSize[0], pos[0])),
                max(0,min(self.screenSize[1], pos[1])))

def get_face(detector, threshold, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image, threshold)
        box_queue.put(box)


def main():
    screen = Screen((3840,2400))

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--draw-markers", action="store_true", default=False,
                    help="")
    ap.add_argument("-c", "--draw-confidence", action="store_true", default=False,
                    help="")
    ap.add_argument("-t", "--confidence-threshold", type=float, default=0.9,
                    help="")
    ap.add_argument("-p", "--draw-pose", action="store_false", default=True,
                    help="")
    ap.add_argument("-u", "--draw-unstable", action="store_true", default=False,
                    help="")
    ap.add_argument("-s", "--draw-segmented", action="store_true", default=False,
                    help="")
    args = vars(ap.parse_args())

    confidence_threshold = args["confidence_threshold"]

    """MAIN"""
    # Video source from webcam or video file.
    video_src = 0
    cam = cv2.VideoCapture(video_src)
    _, sample_frame = cam.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Introduce mark_detector to detect landmarks.
    gaze_detector = GazeEstimator()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    #img_queue.put(sample_frame)

    if isWindows():
        thread = threading.Thread(target=get_face, args=(mark_detector, confidence_threshold, img_queue, box_queue))
        thread.daemon = True
        thread.start()
    else:
        box_process = Process(target=get_face,
                              args=(mark_detector, confidence_threshold, img_queue, box_queue))
        box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cam.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        result = box_queue.get()

        if result is not None:
            if args["draw_confidence"]:
                mark_detector.face_detector.draw_result(frame, result)
            # unpack result
            facebox, confidence = result
            # fix facebox if needed
            if facebox[1] > facebox[3]:
                facebox[1] = 0
            if facebox[0] > facebox[2]:
                facebox[0] = 0
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = mark_detector.detect_marks(face_img)

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # segment the image based on markers and facebox
            seg = Segmenter(facebox, marks, frame.shape[1], frame.shape[0])
            if args["draw_segmented"]:
                mark_detector.draw_box(frame, seg.getSegmentBBs())
                cv2.imshow("fg", seg.getSegmentJSON()["faceGrid"])

            if args["draw_markers"]:
                mark_detector.draw_marks(
                    frame, marks, color=(0, 255, 0))

            # detect gaze
            segments = seg.getSegmentJSON()
            gaze = gaze_detector.detect_gaze(
                frame,
                segments["leftEye"],
                segments["rightEye"],
                segments["face"],
                segments["faceGrid"]
            )
            gaze[0] = -gaze[0]
            print(gaze)
            x,y = screen.cm2Px(gaze)
            #print((x,y))
            pyautogui.moveTo(x,y)

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            stable_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                stable_pose.append(ps_stb.state[0])
            stable_pose = np.reshape(stable_pose, (-1, 3))

            if args["draw_unstable"]:
                pose_estimator.draw_annotation_box(
                    frame, pose[0], pose[1], color=(255, 128, 128))

            if args["draw_pose"]:
                pose_estimator.draw_annotation_box(
                    frame, stable_pose[0], stable_pose[1], color=(128, 255, 128))

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            break

    # Clean up the multiprocessing process.
    if not isWindows():
        box_process.terminate()
        box_process.join()


if __name__ == '__main__':
    main()
