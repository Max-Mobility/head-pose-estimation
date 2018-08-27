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

    def makeBB(self, kp, px=0, py=0):
        x = [x[0] for x in kp]
        y = [x[1] for x in kp]
        bbox = max(np.min(x) - px, 0), max(np.min(y) - py, 0), min(np.max(x) + px, self.width), min(np.max(y) + py, self.height)
        return MarkDetector.get_square_box([int(x) for x in bbox])

    def leftEyeBB(self):
        return self.makeBB(self.marks[42:48], 10, 0)

    def rightEyeBB(self):
        return self.makeBB(self.marks[36:42], 10, 0)

    def faceGrid(self):
        bb = self.faceBB
        x = int(bb[0])
        y = int(bb[2])
        w = int(bb[1] - bb[0])
        h = int(bb[3] - bb[2])
        #Create array of zeros
        faceGrid = np.zeros((self.height, self.width))

        xBound = int(x+w)
        yBound = int(y+h)

        if(x < 0):
            x = 0
        if(y < 0):
            y = 0
        if(xBound > self.width):
            xBound = self.width
        if(yBound > self.height):
            yBound = self.height

        for i in range(x,xBound):
            for j in range(y,yBound):
                faceGrid[j][i] = 1
        return faceGrid

    def segment(self):
        return [
            self.leftEyeBB(),
            self.rightEyeBB(),
            self.faceBB
        ]
        '''
        return {
            'leftEye': self.leftEyeBB(),
            'rightEye': self.rightEyeBB(),
            'face': self.faceBB(),
            'faceGrid': self.faceGrid()
        }
        '''

def main():
    import argparse
    from os_detector import detect_os, isWindows
    # multiprocessing may not work on Windows and macOS, check OS for safety.
    detect_os()

    from multiprocessing import Process, Queue
    import threading

    from pose_estimator import PoseEstimator

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image-folder", type=str, default=".",
                    help="Folder where images are stored")
    args = vars(ap.parse_args())


if __name__ == '__main__':
    main()
