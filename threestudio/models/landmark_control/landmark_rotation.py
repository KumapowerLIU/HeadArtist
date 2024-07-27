import numpy as np
import mediapipe as mp
import cv2
import numpy
import torch

from .face_geometry import get_metric_landmarks, PCF, canonical_metric_landmarks, procrustes_landmark_basis
from PIL import Image
import transforms3d as tfs
from mediapipe.framework.formats import landmark_pb2
import math
from typing import Mapping
from pytorch3d.renderer import (
    PerspectiveCameras
)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)
points_idx = [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()
# points_idx = list(range(0,468)); points_idx[0:2] = points_idx[0:2:-1];

frame_height, frame_width, channels = (512, 512, 3)

# pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

dist_coeff = np.zeros((4, 1))

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
PoseLandmark = mp.solutions.drawing_styles.PoseLandmark

f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for edge in mp_face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw

face_connection_spec_head = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec_head[edge] = head_draw

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5)
class LandmarkRotation():
    def __init__(self, input_image, image_height, image_width, device):

        img = Image.open(input_image).convert("RGB")
        img = img.resize((image_height, image_width))
        img_rgb = numpy.asarray(img)

        results = face_mesh.process(img_rgb)
        face_landmarks = results.multi_face_landmarks[0]
        face_landmark = face_landmarks.landmark

        self.pcf = PCF(near=1, far=10000, frame_height=frame_height, frame_width=frame_width, fy=camera_matrix[1, 1])
        self.landmark = face_landmark
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
    def draw_pupils(self, image, landmark_list, drawing_spec, halfwidth: int = 2):
        """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
        landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
        if len(image.shape) != 3:
            raise ValueError("Input image must be H,W,C.")
        image_rows, image_cols, image_channels = image.shape
        if image_channels != 3:  # BGR channels
            raise ValueError('Input image must contain three channel bgr data.')
        for idx, landmark in enumerate(landmark_list.landmark):
            if (
                (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                (landmark.HasField('presence') and landmark.presence < 0.5)
            ):
                continue
            if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
                continue
            image_x = int(image_cols * landmark.x)
            image_y = int(image_rows * landmark.y)
            draw_color = None
            if isinstance(drawing_spec, Mapping):
                if drawing_spec.get(idx) is None:
                    continue
                else:
                    draw_color = drawing_spec[idx].color
            elif isinstance(drawing_spec, DrawingSpec):
                draw_color = drawing_spec.color
            image[image_y - halfwidth:image_y + halfwidth, image_x - halfwidth:image_x + halfwidth, :] = draw_color

    def pixel_transfer(self, position, size):
        """Converts normalized value pair to pixel coordinates."""
        p = position / size
        return p

    def reverse_channels(self, image):
        """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
        # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
        # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
        return image[:, :, ::-1]

    def pytorch3d_rendering(self, R, t, model_points, f):
        RR = torch.from_numpy(R).permute(1, 0).unsqueeze(0)  # dim = (1, 3, 3)
        tt = torch.from_numpy(t).permute(1, 0)  # dim = (1, 3)
        f = torch.tensor((f, f), dtype=torch.float32).unsqueeze(0)  # dim = (1, 2)
        p = torch.tensor((256, 256), dtype=torch.float32).unsqueeze(0)  # dim = (1, 2)
        img_size = (512, 512)  # (width, height) of the image
        camera = PerspectiveCameras(R=RR, T=tt, focal_length=-f, principal_point=p, image_size=(img_size,),
                                    device=self.device, in_ndc=False)
        out_screen = camera.transform_points_screen(model_points.unsqueeze(0))
        return out_screen[:2].unsqueeze(0).cpu().detach().numpy()

    def changing(self, angle_x, angle_y, angle_z, f, is_only_head):
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in self.landmark])
        landmarks = landmarks.T
        metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), self.pcf)
        model_points = metric_landmarks[0:3, :].T
        rotation_matrix = tfs.euler.euler2mat(angle_x, angle_y, angle_z, 'sxyz')
        mp_rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        mp_translation_vector = pose_transform_mat[:3, 3, None]
        model_points_tensor = torch.from_numpy(model_points).to(self.device)
        model_points_tensor[:, 0] = -model_points_tensor[:, 0]
        out_pt3d = self.pytorch3d_rendering(rotation_matrix, mp_translation_vector, model_points_tensor.float(), f)
        new_landmark = out_pt3d.squeeze()
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=self.pixel_transfer(landmark[0], self.image_width),
                                            y=self.pixel_transfer(landmark[1], self.image_height), z=0) for landmark in
            new_landmark
        ])

        empty = numpy.zeros((self.image_width, self.image_height, 3))
        if is_only_head:
            mp_drawing.draw_landmarks(
                empty,
                face_landmarks_proto,
                connections=face_connection_spec_head.keys(),
                landmark_drawing_spec=None,
                connection_drawing_spec=face_connection_spec_head
            )
        else:
            mp_drawing.draw_landmarks(
                empty,
                face_landmarks_proto,
                connections=face_connection_spec.keys(),
                landmark_drawing_spec=None,
                connection_drawing_spec=face_connection_spec
            )
            # draw eyes
            left_eye = [
                self.pixel_transfer((new_landmark[263][0] + new_landmark[362][0]) / 2, self.image_width) * self.image_width,
                self.pixel_transfer((new_landmark[374][1] + new_landmark[386][1]) / 2,
                                    self.image_height) * self.image_height]
            right_eye = [
                self.pixel_transfer((new_landmark[33][0] + new_landmark[133][0]) / 2, self.image_width) * self.image_width,
                self.pixel_transfer((new_landmark[159][1] + new_landmark[145][1]) / 2,
                                    self.image_height) * self.image_height]
            empty[int(left_eye[1]) - 2:int(left_eye[1]) + 2, int(left_eye[0]) - 2:int(left_eye[0]) + 2,
            :] = left_iris_draw.color
            empty[int(right_eye[1]) - 2:int(right_eye[1]) + 2, int(right_eye[0]) - 2:int(right_eye[0]) + 2,
            :] = right_iris_draw.color
        # Flip BGR back to RGB.
        empty = self.reverse_channels(empty)
        return empty


if __name__ == '__main__':
    input_image = '/home/liuhongyu/code/ControlNetMediaPipeFace/img00000025.png'
    landmark_change = LandmarkRotation(input_image, 512, 512, device='cuda:1')
    out = landmark_change.changing(0, 0, 0, 365, is_only_head=False)
    Image.fromarray(np.uint8(out)).save('/home/liuhongyu/code/ControlNetMediaPipeFace/prior_out/landmark_test.png')
