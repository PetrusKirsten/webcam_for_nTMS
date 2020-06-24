import cv2
import cv2.aruco as aruco
import numpy as np
import FilterforPE

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
translate_tooltip = np.array([0.059, 0.241, -0.005])  # Array to tooltip transformation
translate_arrow = np.array([0.059, 0.241, -0.005])  # Array to probe arrow transformation


def main():
    # Window size for Saviztky-Golay filter
    pose_window = FilterforPE.SavGol(13)
    coil_window = FilterforPE.SavGol(5)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to connect to camera.")
        return

    while cap.isOpened():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Turns the operating frame in gray

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
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        aruco.refineDetectedMarkers(gray, board_probe, corners, ids, rejectedImgPoints)

        if np.all(ids is not None):
            if ids[0] == 0 or ids[0] == 1:  # If detect the probe markers
                # Estimate pose of each marker and return the values rvet and tvec---different from camera coefficient
                retval, rvec, tvec = aruco.estimatePoseBoard(
                    corners, ids, board_probe, cam_matrix,
                    dist_coeffs, rvec=None, tvec=None)

                tvec = np.float32(np.vstack(tvec))
                if retval:
                    aruco.drawAxis(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.1)
                    aruco.drawDetectedMarkers(frame, corners)  # Draw a square around the markers

                    # Mathematical transformations for calculating the pose of the probe
                    rotation_mat = np.identity(4).astype(np.float32)
                    rotation_mat[:3, :3], _ = cv2.Rodrigues(rvec)
                    pose_mat = cv2.hconcat((rotation_mat[:3, :3], tvec))
                    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
                    tool_tip_position = np.dot(rotation_mat[:3, :3],
                                               np.transpose(translate_tooltip)) + np.transpose(tvec)
                    tooltip_mat = cv2.hconcat((rotation_mat[:3, :3], np.transpose(np.float32(tool_tip_position))))
                    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(tooltip_mat[:3, :])
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
                    # The Savitzky-Golay transformations and filter for coil pose
                    rvec, tvec, coil_window.array = FilterforPE.SavGol.filter(
                        coil_window.window,
                        coil_window.array,
                        rvec,
                        tvec)

                    FilterforPE.DrawInfo.coil_coordinates(frame, rvec, tvec)

        cv2.imshow("ArUco Tracker with Savitzky-Golay Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    main()
