import cv2
import itertools
import numpy as np
import time
import mediapipe as mp
import re
from preprocesses import *
import torch
import glob
import os
from skimage import transform as transf
from tqdm import tqdm


face_oval_avgs = np.load('/home/taylorpap/Bootcamp/face_oval_averages.npz', allow_pickle=True)['data']
face_oval_avgs = face_oval_avgs * 256
std_size = (256, 256)
landmark_indexes_for_cropping = [2, 3, 10, 11, 26, 30]


class VideoPreprocessor(object):
    def __init__(self):

        self.init_crop_width = 96
        self.init_crop_height = 96
        self.actual_crop_size = (88, 88)


    def extract_points_from_mesh(self, face_landmarks, indexes):
        points_data_regex = re.compile(r'\d\.\d+')
        xy_points_list = []
        for count, each_index in enumerate(indexes):
            xyzpointsraw = face_landmarks.landmark[each_index]
            points_list = points_data_regex.findall(str(xyzpointsraw))
            if len(points_list) < 1:
                xy_points_list.append([None])
            else:
                xyclean = [float(points_list[0]), float(points_list[1])]
                xy_points_list.append(xyclean)
        xy_points_array = np.array(xy_points_list)
        return xy_points_array

    def detect_Facial_Landmarks(self, image):
        face_mesh_videos, mp_face_mesh = self.mediapipe_setup()
        frame_dict = {}
        mesh_result = face_mesh_videos.process(image)
        oval_indexes = list(set(itertools.chain(*mp_face_mesh.FACEMESH_FACE_OVAL)))
        lips_indexes = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
        if mesh_result.multi_face_landmarks:
            for face_no, face_landmarks in enumerate(mesh_result.multi_face_landmarks):
                oval_points_array = self.extract_points_from_mesh(face_landmarks, oval_indexes)
                lips_points_array = self.extract_points_from_mesh(face_landmarks, lips_indexes)
                frame_dict['oval_landmarks'] = oval_points_array
                frame_dict['lips_landmarks'] = lips_points_array
        else:
            frame_dict['oval_landmarks'] = None
            frame_dict['lips_landmarks'] = None
        return frame_dict

    def mediapipe_setup(self):
        # Initialize the mediapipe face detection class
        mp_face_detection = mp.solutions.face_detection
        # Setup the face detection function
        face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        # initialize the mediapipe face mesh class
        mp_face_mesh = mp.solutions.face_mesh
        # Setup the face landmarks function for videos
        face_mesh_videos = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                 min_detection_confidence=0.5, min_tracking_confidence=0.3)
        return face_mesh_videos, mp_face_mesh

    def get_face_points(self, video):
        vid_capture = cv2.VideoCapture(video)
        frame_idx = 0
        sequence = []
        if (vid_capture.isOpened() == False):
            print("Error opening the video file")
        else:
            while (vid_capture.isOpened()):
                ret, frame = vid_capture.read()
                if ret:
                    frame_points_dict = self.detect_Facial_Landmarks(frame)
                    current_oval = frame_points_dict['oval_landmarks'] * 256
                    current_lips = frame_points_dict['lips_landmarks']
                    transformed_frame, trans_mat = self.warp_img(current_oval[landmark_indexes_for_cropping, :],
                                                            face_oval_avgs[landmark_indexes_for_cropping, :],
                                                            frame,
                                                            std_size)
                    trans_lips = trans_mat(current_lips * 256)
                    cut_frame = self.crop_out_patch(transformed_frame, trans_lips, self.init_crop_height // 2, self.init_crop_width // 2)
                    sequence.append(cut_frame)
                    #all_points.append(frame_points_dict)
                    frame_idx += 1
                else:
                    break
        vid_capture.release()
        cv2.destroyAllWindows()
        return self.convert_bgr2gray(np.array(sequence))

    # Create Method for warping image and getting transform parameters
    def warp_img(self, src, dst, img, std_size):
        tform = transf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        warped = transf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # wrap the frame image
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped, tform

    # Create Method to apply a previously calculated transform
    def apply_transform(self, transform, img, std_size):
        warped = transf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')
        return warped

    def crop_out_patch(self, img, landmarks, height, width):
        center_x, center_y = np.mean(landmarks, axis=0)

        cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                             int(round(center_x) - round(width)): int(round(center_x) + round(width))])
        return cutted_img

    def convert_bgr2gray(self, data):
        return np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in data], axis=0)

    def preprocess_creation(self):
        crop_size = self.actual_crop_size
        (mean, std) = (0.421, 0.165)
        preprocessed = Compose([
            Normalize(0.0, 255.0),
            CenterCrop(crop_size),
            Normalize(mean, std)])
        return preprocessed

    def full_tensor_setup(self, video):
        video_array = self.get_face_points(video)
        preprocess_video = self.preprocess_creation()
        video_tensor = torch.Tensor(video_array)
        preprocessed_tensor = preprocess_video(video_tensor)
        return preprocessed_tensor.unsqueeze(0), len(video_array)

