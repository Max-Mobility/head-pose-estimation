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


def get_face(height, width, threshold, img_queue, box_queue):
    import queue as Q
    from multiprocessing import Process, Queue

    import numpy as np

    import cv2
    from mark_detector import MarkDetector
    from pose_estimator import PoseEstimator
    from stabilizer import Stabilizer
    from gaze_estimator import GazeEstimator

    from segmenter import Segmenter

    """Get face from image queue. This function is used for multiprocessing"""

    CNN_INPUT_SIZE = 128

    # Introduce mark_detector to detect landmarks.
    md = MarkDetector()

    # Introduce mark_detector to detect landmarks.
    gd = GazeEstimator()

    pe = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    while True:
        result = None
        image = None
        try:
            image = img_queue.get(timeout=1)
        except Q.Empty as inst:
            tmp = None
            #print("no image received")
        if image is not None:
            result = md.extract_cnn_facebox(image, threshold)
            #box_queue.put(result)

        if result is not None:
            # unpack result
            facebox, confidence = result
            # fix facebox if needed
            if facebox[1] > facebox[3]:
                facebox[1] = 0
            if facebox[0] > facebox[2]:
                facebox[0] = 0
            # Detect landmarks from image of 128x128.
            face_img = image[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = md.detect_marks(face_img)
            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # segment the image based on markers and facebox
            seg = Segmenter(facebox, marks, image.shape[1], image.shape[0])

            # detect gaze
            segments = seg.getSegmentJSON()
            gaze = gd.detect_gaze(
                image,
                segments["leftEye"],
                segments["rightEye"],
                segments["face"],
                segments["faceGrid"]
            )

            # Try pose estimation with 68 points.
            pose = pe.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            stable_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                stable_pose.append(ps_stb.state[0])
            stable_pose = np.reshape(stable_pose, (-1, 3))

            box_queue.put((image, facebox, confidence, marks, segments, gaze, pose, stable_pose))
        else:
            box_queue.put(None)

def main():
    from multiprocessing import Process, Queue
    import threading

    import argparse

    import cv2

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

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    #img_queue.put(sample_frame)

    num_threads = 5
    tids = []
    if isWindows():
        for i in range(num_threads):
            thread = threading.Thread(target=get_face, args=(height, width, confidence_threshold, img_queue, box_queue))
            thread.daemon = True
            thread.start()
            tids.append(thread)
    else:
        for i in range(num_threads):
            box_process = Process(target=get_face,
                                  args=(height, width, confidence_threshold, img_queue, box_queue))
            box_process.start()
            tids.append(box_process)


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
            # unpack result
            (image, facebox, confidence, marks, segments, gaze, pose, stable_pose) = result

            print(gaze)

            # Show preview.
            cv2.imshow("Preview", image)

        # await exit
        if cv2.waitKey(10) == 27:
            break

    # Clean up the multiprocessing process.
    if not isWindows():
        for tid in tids:
            tid.terminate()
            tid.join()


if __name__ == '__main__':
    main()
