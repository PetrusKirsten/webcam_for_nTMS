import cv2
import cv2.aruco as aruco
import dlib
import numpy as np
from imutils import face_utils
import FilterforPE
from FilterforPE import stabilizer

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
translate_tooltip = np.array([0.059, 0.241, -0.005])  # Array to tooltip transformation
translate_arrow = np.array([0.059, 0.241, -0.005])  # Array to probe arrow transformation
# Kalman filter coefficients
ref_stabilizers = [stabilizer.KalmanFace(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=2) for _ in range(6)]
probe_stabilizers = [stabilizer.KalmanProbe(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=5) for _ in range(6)]
obj_stabilizers = [stabilizer.KalmanCoil(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=5) for _ in range(6)]


def get_head_pose(shape):
    kalman_on = False
    # Set the face landmarks of interest
    points_2D = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36], shape[39], shape[42],
                            shape[45], shape[31], shape[35], shape[48], shape[54], shape[57], shape[8]])
    # Determine the 3D head pose
    _, rotation_vec, translation_vector = cv2.solvePnP(points_3D, points_2D, cam_matrix, dist_coeffs,
                                                       flags=cv2.SOLVEPNP_ITERATIVE)
    # Calculate euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    # Kalman filter
    if kalman_on:
        kalman_array = []
        pose_np = np.array((euler_angle, translation_vector)).flatten()
        for value, ps_stb in zip(pose_np, ref_stabilizers):
            ps_stb.update([value])
            kalman_array.append(ps_stb.state[0])
        kalman_array = np.reshape(kalman_array, (-1, 3))
        euler_angle, translation_vector = kalman_array

    euler_angle = np.reshape(euler_angle, (3, 1))
    translation_vector = np.reshape(translation_vector, (3, 1))

    return euler_angle, translation_vector


def main():
    # Window size for Saviztky-Golay filter
    face_window = FilterforPE.SavGol(31)
    pose_window = FilterforPE.SavGol(13)
    coil_window = FilterforPE.SavGol(7)

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
            # The Savitzky-Golay transformations and filter for head pose
            orientation, position, face_window.array = FilterforPE.SavGol.filter(
                face_window.window,
                face_window.array,
                orientation,
                position)
            # Determine the head axis origin
            reproject_distance, _ = cv2.projectPoints(
                face_box_size,
                np.deg2rad(orientation),
                position,
                cam_matrix,
                dist_coeffs)

            # Set the threshold to avoid the algorithm to collapse due a bad camera angle
            min_threshold = 100
            max_threshold = 600
            while np.amax(reproject_distance) > max_threshold or np.amin(reproject_distance) < min_threshold:
                orientation, position = get_head_pose(shape)
                # Determine the head axis origin without filter (avoinding collapse)
                reproject_distance, _ = cv2.projectPoints(
                    face_box_size,
                    np.deg2rad(orientation),
                    position,
                    cam_matrix,
                    dist_coeffs)
                # Advice message
                cv2.rectangle(frame, (10, 110), (200, 150), (0, 255, 255), -1)
                cv2.rectangle(frame, (10, 110), (200, 150), (0, 0, 255), 1)
                cv2.putText(frame, "* ADJUST THE HEAD POSE *", (15, 134),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), thickness=1)
                break

            if np.amax(reproject_distance) < max_threshold and np.amin(reproject_distance) > min_threshold:
                reproject_distance = tuple(map(tuple, reproject_distance.reshape(8, 2)))

                FilterforPE.DrawInfo.face_box(frame, reproject_distance, shape)
                FilterforPE.DrawInfo.face_coordinates(frame, orientation, position)

        # ArUco operations
        markerLength = 0.05  # Here, our measurement unit is centimetre.
        markerSeparation = 0.005  # Here, our measurement unit is centimetre.
        # Build the ArUco Marker's dictionary and parameters
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        parameters.cornerRefinementWinSize = 5
        # Create the markers configuration
        board_probe = aruco.GridBoard_create(2, 1, markerLength, markerSeparation, aruco_dict, firstMarker=0)
        board_coil = aruco.GridBoard_create(2, 1, markerLength, markerSeparation, aruco_dict, firstMarker=2)
        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)
        aruco.refineDetectedMarkers(gray_frame, board_probe, corners, ids, rejectedImgPoints)

        if np.all(ids is not None):
            if ids[0] == 0 or ids[0] == 1:  # If detect the probe markers
                # Estimate pose of each marker and return the values rvet and tvec---different from camera coefficient
                retval, rvec, tvec = aruco.estimatePoseBoard(
                    corners, ids, board_probe, cam_matrix,
                    dist_coeffs, rvec=None, tvec=None)

                tvec = np.float32(np.vstack(tvec))
                if retval:
                    aruco.drawAxis(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.1)
                    aruco.drawDetectedMarkers(frame, corners)

                    # Mathematical transformations for calculating the pose of the probe
                    rotation_mat = np.identity(4).astype(np.float32)
                    rotation_mat[:3, :3], _ = cv2.Rodrigues(rvec)
                    pose_mat = cv2.hconcat((rotation_mat[:3, :3], tvec))
                    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
                    tool_tip_position = np.dot(rotation_mat[:3, :3],
                                               np.transpose(translate_tooltip)) + np.transpose(tvec)
                    tooltip_mat = cv2.hconcat((rotation_mat[:3, :3], np.transpose(np.float32(tool_tip_position))))
                    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(tooltip_mat[:3, :])

                    # Kalman filter for probe pose
                    kalman_array = []
                    pose_np = np.array((euler_angle, np.transpose(tool_tip_position))).flatten()
                    for value, ps_stb in zip(pose_np, probe_stabilizers):
                        ps_stb.update([value])
                        kalman_array.append(ps_stb.state[0])
                    kalman_array = np.reshape(kalman_array, (-1, 3))
                    euler_angle, tool_tip_position = kalman_array
                    euler_angle = np.reshape(euler_angle, (3, 1))
                    tool_tip_position = np.reshape(tool_tip_position, (3, 1))

                    # The Savitzky-Golay transformations and filter for probe pose
                    euler_angle, tool_tip_position, pose_window.array = FilterforPE.SavGol.filter(
                        pose_window.window,
                        pose_window.array,
                        euler_angle,
                        tool_tip_position)

                    reprojectdst, _ = cv2.projectPoints(  # Determine the probe origin
                        np.float32([[0, 0, 0]]),
                        np.deg2rad(euler_angle),
                        tvec,
                        cam_matrix,
                        dist_coeffs)
                    reprojectdst_arrow, _ = cv2.projectPoints(  # Determine the probe indication
                        np.float32([[0, 0, 0]]),
                        np.deg2rad(euler_angle),
                        tool_tip_position,
                        cam_matrix,
                        dist_coeffs)

                    FilterforPE.DrawInfo.probe_arrow(frame, reprojectdst, reprojectdst_arrow)
                    FilterforPE.DrawInfo.probe_coordinates(frame, euler_angle, tool_tip_position)

            if ids[0] == 2 or ids[0] == 3:  # If detect the coil markers
                # Estimate pose of each marker and return the values rvet and tvec---different from camera coefficient
                retval, rvec, tvec = aruco.estimatePoseBoard(
                    corners,
                    ids,
                    board_coil,
                    cam_matrix,
                    dist_coeffs,
                    rvec=None,
                    tvec=None)
                tvec = np.float32(np.hstack(tvec))

                if retval:
                    aruco.drawDetectedMarkers(frame, corners)  # Draw a square around the markers
                    aruco.drawAxis(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.1)

                    tvec = np.reshape(tvec, (3, 1))
                    # Kalman filter for coil pose
                    kalman_array = []
                    pose_np = np.array((rvec, tvec)).flatten()
                    for value, ps_stb in zip(pose_np, probe_stabilizers):
                        ps_stb.update([value])
                        kalman_array.append(ps_stb.state[0])
                    kalman_array = np.reshape(kalman_array, (-1, 3))
                    rvec, tvec = kalman_array

                    # The Savitzky-Golay transformations and filter for coil pose
                    rvec, tvec, coil_window.array = FilterforPE.SavGol.filter(
                        coil_window.window,
                        coil_window.array,
                        rvec,
                        tvec)

                    FilterforPE.DrawInfo.coil_coordinates(frame, rvec, tvec)

        cv2.imshow("HPE with Savitzky-Golay Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    main()
