import cv2
import dlib
import numpy as np
from imutils import face_utils
from FilterforPE import DrawInfo

# Some coefficients for camera, image and probe calculation
K = [6.5308391993466671e+002,
     0.0,
     3.1950000000000000e+002,
     0.0,
     6.5308391993466671e+002,
     2.3950000000000000e+002,
     0.0,
     0.0,
     1.0]
D = [7.0834633684407095e-002,
     6.9140193737175351e-002,
     0.0,
     0.0,
     -1.3073460323689292e+000]
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
points_3D = np.float32([[6.825897, 6.760612, 4.402142],
                        [1.330353, 7.122144, 6.903745],
                        [-1.330353, 7.122144, 6.903745],
                        [-6.825897, 6.760612, 4.402142],
                        [5.311432, 5.485328, 3.987654],
                        [1.789930, 5.393625, 4.413414],
                        [-1.789930, 5.393625, 4.413414],
                        [-5.311432, 5.485328, 3.987654],
                        [2.005628, 1.409845, 6.165652],
                        [-2.005628, 1.409845, 6.165652],
                        [2.774015, -2.080775, 5.048531],
                        [-2.774015, -2.080775, 5.048531],
                        [0.000000, -3.116408, 6.097667],
                        [0.000000, -7.415691, 4.070434]])  # Set the face 3D points localization
face_box_size = np.float32([[10.0, 10.0, 10.0],
                            [10.0, 10.0, -10.0],
                            [10.0, -10.0, -10.0],
                            [10.0, -10.0, 10.0],
                            [-10.0, 10.0, 10.0],
                            [-10.0, 10.0, -10.0],
                            [-10.0, -10.0, -10.0],
                            [-10.0, -10.0, 10.0]])  # Set the size of axis box


def get_head_pose(shape):
    # Set the face landmarks of interest
    points_2D = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36], shape[39], shape[42],
                            shape[45], shape[31], shape[35], shape[48], shape[54], shape[57], shape[8]])
    # Determine the 3D head pose
    _, rotation_vector, translation_vector = cv2.solvePnP(
        points_3D,
        points_2D,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)
    # Calculate euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return euler_angle, translation_vector


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to connect to camera.")
        return

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'Predictors\shape_predictor_68_face_landmarks.dat')

    while cap.isOpened():
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Turns the operating frame in gray
        face_rects = detector(gray_frame, 0)  # Detect the face in the image frame

        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)

            orientation, position = get_head_pose(shape)

            # Determine the head axis origin
            reproject_distance, _ = cv2.projectPoints(
                face_box_size,
                np.deg2rad(orientation),
                position,
                cam_matrix,
                dist_coeffs)
            reproject_distance = tuple(map(tuple, reproject_distance.reshape(8, 2)))

            DrawInfo.face_box(frame, reproject_distance, shape)
            DrawInfo.face_coordinates(frame, orientation, position)

        cv2.imshow("Head Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    main()
