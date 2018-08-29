import cv2
import tensorflow as tf
import numpy as np

class GazeEstimator:

    def __init__(self, gaze_model='model/mobileNet.pb', eye_image_size=224, face_image_size=224):
        self.eye_image_size = eye_image_size
        self.face_image_size = face_image_size
        # Get a TensorFlow session ready to do landmark detection
        # Load a (frozen) Tensorflow model into memory.
        gaze_graph = tf.Graph()
        with gaze_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(gaze_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = gaze_graph
        self.sess = tf.Session(graph=gaze_graph)

    @staticmethod
    def crop(image, box):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    @staticmethod
    def resize(image, size):
        return cv2.resize(image, (size, size))

    @staticmethod
    def normalize(image, maxVal):
        return np.divide(image, maxVal)

    def detect_gaze(self, image_np, left_eye_bb, right_eye_bb, face_bb, face_grid):
        # crop input image into sections by bounding boxes
        leftEye = self.crop(image_np, left_eye_bb)
        rightEye = self.crop(image_np, left_eye_bb)
        face = self.crop(image_np, face_bb)
        # normalize input image into range [0,1]
        leftEye = self.normalize(leftEye, 255)
        rightEye = self.normalize(rightEye, 255)
        face = self.normalize(face, 255)
        # resize input image
        leftEye = self.resize(leftEye, self.eye_image_size)
        rightEye = self.resize(rightEye, self.eye_image_size)
        face = self.resize(face, self.face_image_size)
        faceGrid = self.resize(face_grid, 25)
        faceGrid = np.reshape(faceGrid, 625)

        # get output tensor by name
        output_tensor = self.graph.get_tensor_by_name('output_node00:0')

        # now actually run prediction
        prediction = self.sess.run(
            output_tensor,
            feed_dict={
                'input_1:0': [leftEye],
                'input_2:0': [rightEye],
                'input_3:0': [face],
                'input_4:0': [faceGrid]
            })

        # convert prediction to gaze
        gaze = np.array(prediction).flatten()

        return gaze
