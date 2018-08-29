import os
from os_detector import detect_os, isWindows
detect_os()

import json
import numpy as np
import cv2
from mark_detector import MarkDetector

# starts in left (right on person) corner and goes clockwise
reIndices = [37,38,39,40,41,42]
leIndices = [43,44,45,46,47,48]

class Segmenter:
    def __init__(self, faceBox, marks, width, height):
        self.marks = marks
        self.width = width
        self.height = height
        # face bb:
        size = faceBox[2] - faceBox[0]
        # offsets
        xo = 0
        yo = int(-size * 0.05)
        # padding
        px = 0
        py = int(size * 0.05)
        #cropping
        cropRatio = 0.8
        diff = int(size * (1.0-cropRatio) / 2.0)
        # transform faceBox
        self.faceBB = [
            faceBox[0] + diff - px + xo,
            faceBox[1] + diff - py + yo,
            faceBox[2] - diff + px + xo,
            faceBox[3] - diff + py + yo
        ]
        self.leBB = self.getLeftEyeBB()
        self.reBB = self.getRightEyeBB()
        self.faceGrid = self.getFaceGrid()

    def makeBB(self, kp, px=0, py=0):
        x = [x[0] for x in kp]
        y = [x[1] for x in kp]
        bbox = max(np.min(x) - px, 0), max(np.min(y) - py, 0), min(np.max(x) + px, self.width), min(np.max(y) + py, self.height)
        return MarkDetector.get_square_box([int(x) for x in bbox], [self.height, self.width])

    def getLeftEyeBB(self):
        return self.makeBB(self.marks[42:48], 10, 0)

    def getRightEyeBB(self):
        return self.makeBB(self.marks[36:42], 10, 0)

    def getFaceGrid(self):
        # make sure the facegrid is square
        # (pad with zeroes on each side)
        size = max(self.height, self.width)
        #Create array of zeros
        faceGrid = np.zeros((size, size))
        diff = self.height - self.width
        ox = 0
        oy = 0
        # compute offsets from squaring
        if diff > 0: # height > width
            ox = int(abs(diff) / 2)
        elif diff < 0: # height < width
            oy = int(abs(diff) / 2)
        # get the face bounding box
        bb = self.faceBB
        # make sure to use any offsets from making the image square
        x = int(bb[0] + ox)
        y = int(bb[1] + oy)
        w = int(bb[2] - bb[0])
        h = int(bb[3] - bb[1])

        xBound = int(x+w)
        yBound = int(y+h)

        if(x < 0):
            x = 0
        if(y < 0):
            y = 0
        if(xBound > size):
            xBound = size
        if(yBound > size):
            yBound = size

        # set faceGrid bounding box (in 25x25 shape)
        factor = 25 / size
        self.faceGridBB = [
            int(x * factor), int(y * factor),
            int(xBound * factor), int(yBound * factor)
        ]

        for i in range(x,xBound):
            for j in range(y,yBound):
                faceGrid[j][i] = 1
        return faceGrid

    def getSegmentJSON(self):
        return {
            'leftEye': self.leBB,
            'rightEye': self.reBB,
            'face': self.faceBB,
            'faceGrid': self.faceGrid,
            'faceGridBB': self.faceGridBB
        }

    def getSegmentBBs(self):
        return [
            self.leBB,
            self.reBB,
            self.faceBB
        ]

class Subject:
    def __init__(self, path):
        self.path = path
        # output json structures
        # these will be read in
        self.framesJSON = {}
        self.dotJSON = {}
        # these will be created
        self.leftEyeJSON = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'isValid': []
        }
        self.rightEyeJSON = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'isValid': []
        }
        self.faceJSON = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'isValid': []
        }
        self.faceGridJSON = {
            'X': [],
            'Y': [],
            'W': [],
            'H': [],
            'isValid': []
        }

    def addSegments(self, index, segmentJSON=None):
        # Note: this function does not update dotJSON or framesJSON -
        #       since they are loaded and should be unchanged
        self.leftEyeJSON['X'].append(0)
        self.leftEyeJSON['Y'].append(0)
        self.leftEyeJSON['W'].append(0)
        self.leftEyeJSON['H'].append(0)
        self.leftEyeJSON['isValid'].append(0)
        self.rightEyeJSON['X'].append(0)
        self.rightEyeJSON['Y'].append(0)
        self.rightEyeJSON['W'].append(0)
        self.rightEyeJSON['H'].append(0)
        self.rightEyeJSON['isValid'].append(0)
        self.faceJSON['X'].append(0)
        self.faceJSON['Y'].append(0)
        self.faceJSON['W'].append(0)
        self.faceJSON['H'].append(0)
        self.faceJSON['isValid'].append(0)
        self.faceGridJSON['X'].append(0)
        self.faceGridJSON['Y'].append(0)
        self.faceGridJSON['W'].append(0)
        self.faceGridJSON['H'].append(0)
        self.faceGridJSON['isValid'].append(0)
        if segmentJSON is not None:
            # update leftEyeJSON
            le = segmentJSON["leftEye"]
            self.leftEyeJSON['X'][index] = le[0]
            self.leftEyeJSON['Y'][index] = le[1]
            self.leftEyeJSON['W'][index] = le[2] - le[0]
            self.leftEyeJSON['H'][index] = le[3] - le[1]
            self.leftEyeJSON['isValid'][index] = 1
            # update rightEyeJSON
            re = segmentJSON["rightEye"]
            self.rightEyeJSON['X'][index] = re[0]
            self.rightEyeJSON['Y'][index] = re[1]
            self.rightEyeJSON['W'][index] = re[2] - re[0]
            self.rightEyeJSON['H'][index] = re[3] - re[1]
            self.rightEyeJSON['isValid'][index] = 1
            # update faceJSON
            f = segmentJSON["face"]
            self.faceJSON['X'][index] = f[0]
            self.faceJSON['Y'][index] = f[1]
            self.faceJSON['W'][index] = f[2] - f[0]
            self.faceJSON['H'][index] = f[3] - f[1]
            self.faceJSON['isValid'][index] = 1
            # update faceGridJSON
            # Note: FG is 1-indexed, so we must add one
            fg = segmentJSON["faceGridBB"]
            self.faceGridJSON['X'][index] = fg[0] + 1
            self.faceGridJSON['Y'][index] = fg[1] + 1
            self.faceGridJSON['W'][index] = fg[2] - fg[0]
            self.faceGridJSON['H'][index] = fg[3] - fg[1]
            self.faceGridJSON['isValid'][index] = 1

    def writeSegmentFiles(self, folder):
        fullDir = self.path + '/' + folder
        # check if the folder exists
        if not os.path.isdir(fullDir):
            os.mkdir(fullDir)
        # write frames.json
        fname = 'frames.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.framesJSON))
        # write appleFace.json
        fname = 'appleFace.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.faceJSON))
        # write appleLeftEye.json
        fname = 'appleLeftEye.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.leftEyeJSON))
        # write appleRightEye.json
        fname = 'appleRightEye.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.rightEyeJSON))
        # write faceGrid.json
        fname = 'faceGrid.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.faceGridJSON))
        # write dotInfo.json
        fname = 'dotInfo.json'
        with open(fullDir + '/' + fname, 'w') as f:
            f.write(json.dumps(self.dotJSON))

    def getFramesJSON(self):
        with open(self.path + '/frames.json') as f:
            frames = json.load(f)
        # update frameJSON to be same as loaded file
        self.framesJSON = frames
        return frames

    def getFaceJSON(self):
        with open(self.path + '/appleFace.json') as f:
            faceMeta = json.load(f)
        return faceMeta

    def getEyesJSON(self):
        with open(self.path + '/appleLeftEye.json') as f:
            leftEyeMeta = json.load(f)
        with open(self.path + '/appleRightEye.json') as f:
            rightEyeMeta = json.load(f)
        return (leftEyeMeta, rightEyeMeta)

    def getFaceGridJSON(self):
        with open(self.path + '/faceGrid.json') as f:
            faceGridMeta = json.load(f)
        return faceGridMeta

    def getDotJSON(self):
        with open(self.path + '/dotInfo.json') as f:
            dotMeta = json.load(f)
        # update dotJSON to be the same as the loaded file
        self.dotJSON = dotMeta
        return dotMeta

    def getImage(self, imagePath):
        return cv2.imread(imagePath)

def main():
    import sys
    from functools import reduce
    import time
    import argparse
    import queue as Q
    from multiprocessing import Process, Queue
    import threading

    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-folder", type=str, default=".",
                    help="Folder containing unzipped subject folders")
    ap.add_argument("-o", "--output-prefix", type=str, default="custom_segmentation",
                    help="Name / Prefix for output folder")
    ap.add_argument("-c", "--use-confidence", action="store_true", default=False,
                    help="Flag to enable use of confidence as 'isValid' metric")
    ap.add_argument("-t", "--confidence-threshold", type=float, default=0.7,
                    help="Number of threads to spawn")
    ap.add_argument("-n", "--num-threads", type=int, default="10",
                    help="Number of threads to spawn")
    args = vars(ap.parse_args())

    # init shared variables
    CNN_INPUT_SIZE = 128
    output_prefix = args["output_prefix"]
    conf_threshold = args["confidence_threshold"]
    use_confidence = args["use_confidence"]

    def process_subject(done_queue, sub_queue):
        """Get subject from subject queue. This function is used for multiprocessing"""
        # init process/thread variables
        detector = MarkDetector()
        timeout = 1 # 1 second timeout waiting for more subjects
        while True:
            try:
                # get subject from queue
                subject = sub_queue.get(timeout=timeout)
            except Q.Empty as inst:
                break;
            subjectPath = subject.path
            subjectID = subjectPath.split('/')[-1]

            print("Processing subject:",subjectID)

            # load MIT metadata
            frameNames = subject.getFramesJSON()
            # Collecting metadata about face, eyes, facegrid, labels
            face = subject.getFaceJSON()
            leftEye, rightEye = subject.getEyesJSON()
            faceGrid = subject.getFaceGridJSON()
            dotInfo = subject.getDotJSON()

            # Iterate over frames for the current subject
            for i, (frame, fv, lv, rv, fgv) in enumerate(zip(frameNames,
                                                             face['IsValid'],
                                                             leftEye['IsValid'],
                                                             rightEye['IsValid'],
                                                             faceGrid['IsValid'])):
                # we'll need to make sure all frames are processed so
                # we must call Subject::addSegments for every frame -
                # it will set isValid to False if segmentJSON is None
                segmentJSON = None
                # Check if cur frame is valid
                if(use_confidence or fv*lv*rv*fgv == 1):
                    # Generate path for frame
                    framePath = subjectPath + "/frames/" + frame
                    # load image data
                    image = subject.getImage(framePath)
                    result = detector.extract_cnn_facebox(image, conf_threshold)

                    if result is not None:
                        # unpack result
                        facebox, confidence = result

                        try:
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
                            marks = detector.detect_marks(face_img)

                            # Convert the marks locations from local CNN to global image.
                            marks *= (facebox[2] - facebox[0])
                            marks[:, 0] += facebox[0]
                            marks[:, 1] += facebox[1]

                            # segment the image based on markers and facebox
                            seg = Segmenter(facebox, marks, image.shape[0], image.shape[1])
                            segmentJSON = seg.getSegmentJSON()
                        except cv2.error as inst:
                            print("Error processing subject:", subjectID,'frame:', i, inst)
                # add segment data to subject
                subject.addSegments(i, segmentJSON)

            # write out the metadata file
            subject.writeSegmentFiles(output_prefix)

            print("Finished processing subject:", subjectID)

        # mark that we're done here!
        print("processing thread done!")
        done_queue.put(True)

    # get directory to subjects
    path = args["input_folder"]
    subjectDirs = os.listdir(path=path)
    num_subjects = len(subjectDirs)
    num_subjects_processed = 0

    # TODO: find better way to control multiprocessing and memory usage
    parallelization = min(args["num_threads"], num_subjects)

    # Setup process and queues for multiprocessing.
    sub_queue = Queue()
    done_queue = Queue()
    # spawn some number of threads / processes here
    tids = []
    if isWindows():
        for tid in range(parallelization):
            thread = threading.Thread(target=process_subject,
                                      args=(done_queue, sub_queue))
            thread.daemon = True
            thread.start()
            tids.append(thread)
    else:
        for tid in range(parallelization):
            box_process = Process(target=process_subject,
                                  args=(done_queue, sub_queue))
            box_process.start()
            tids.append(box_process)


    # TODO: might need better control over memory management
    max_queue_size = parallelization * 2
    while True:
        if sub_queue.qsize() < max_queue_size and num_subjects_processed < num_subjects:
            subDir = subjectDirs[num_subjects_processed]
            subject = Subject(path + '/' + subDir)
            # feed subject into subject queue.
            sub_queue.put(subject)
            # update the number of subjects we have processed
            num_subjects_processed += 1
        else:
            # wait to not take up cpu time
            time.sleep(0.1)
        # are the threads done?
        if done_queue.qsize() >= len(tids):
            print("All threads done, exiting!")
            break;

    # clean up process
    if not isWindows():
        for tid in tids:
            tid.terminate()
            tid.join()


if __name__ == '__main__':
    main()
