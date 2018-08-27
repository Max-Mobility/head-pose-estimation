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

        for i in range(x,xBound):
            for j in range(y,yBound):
                faceGrid[j][i] = 1
        return faceGrid

    def segment(self):
        return {
            'leftEye': self.leBB,
            'rightEye': self.reBB,
            'face': self.faceBB,
            'faceGrid': self.faceGrid
        }

    def getSegmentBBs(self):
        return [
            self.leBB,
            self.reBB,
            self.faceBB
        ]

def main():

    import argparse
    from os_detector import detect_os, isWindows
    detect_os()

    from multiprocessing import Process, Queue
    import threading

    # TODO: update this function to do all the processing and
    #       serialization of new metadata file
    def process_image(img_queue):
        """Get face from image queue. This function is used for multiprocessing"""
        # init variables
        detector = MarkDetector()

        while True:
            # get image from queue
            image = img_queue.get()
            facebox, confidence = detector.extract_cnn_facebox(image)

            # Detect landmarks from image of 128x128.
            face_img = image[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = mark_detector.detect_marks(face_img)

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # segment the image based on markers and facebox
            seg = Segmenter(facebox, marks, image.shape[0], image.shape[1])
            metadata = seg.segment()

            # TODO: write out the metadata into json file

            # TODO: need breaking condition here

    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image-folder", type=str, default=".",
                    help="Folder where images are stored")
    args = vars(ap.parse_args())

    # Setup process and queues for multiprocessing.
    img_queue = Queue()

    # TODO: spawn more than one thread/process for better processing
    # spawn some number of threads / processes here
    if isWindows():
        thread = threading.Thread(target=process_image, args=(mark_detector, img_queue))
        thread.daemon = True
        thread.start()
    else:
        box_process = Process(target=process_image,
                              args=(mark_detector, img_queue))
        box_process.start()

    # TODO: find better way to control multiprocessing and memory usage
    parallelization = 100
    # TODO: might need better control over memory management
    max_queue_size = 0
    # TODO: read number of images from folder
    num_images = 0
    num_images_processed = 0
    while True:
        # TODO: load images from dataset (iteratively, may not fit
        #       everything in memory) - need better performance for this
        if len(img_queue) < max_queue_size:
            # TODO: get image from disk

            # feed image into image queue.
            img_queue.put(frame)

        # TODO: need better way to determine processing is done
        if num_images_processed > num_images:
            break;

    # clean up process
    if not isWindows():
        box_process.terminate()
        box_process.join()


if __name__ == '__main__':
    main()
