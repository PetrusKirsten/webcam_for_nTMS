import cv2
import numpy as np
from scipy.signal import savgol_filter


class SavGol:
    """
    Filters and manipulates pose arrays (3 positions and 3 orientations).

    :param window: int
        This determine the window size.

    """
    def __init__(self, window):
        self.window = window
        self.array = np.zeros(6 * window)


    @staticmethod
    def filter(window, array2transform, orientation, position):
        """
        Contains the Savitzky-Golay filter itself and respective transformations.

        :param window: int
            The window size predetermined.
        :param array2transform: array
            The array with 6 rows (3 positions and 3 orientations) and 'window' columns to be manipulated.
        :param orientation: array
            The array (1 x 3) containing the orientation data.
        :param position: array
            The array (1 x 3) containing the position data.
        :return:
            euler_angle: array
                Filtered array containing the orientation data in euler angle.
            translation_vec: array
                Filtered array containing the position data.
            array2transform: array
                Filtered and manipulated array in window size.

        """

        pose_flatten = np.array((orientation, position)).flatten()
        for element2delete in range(0, 5 * (window - 1) + 1, window - 1):  # Delete the passed elements in each frame
            array2transform = np.delete(array2transform, element2delete)
        element2insert = 0
        for position2insert in range(window - 1, window * 6, window):  # Insert the new elements in each frame
            array2transform = np.insert(array2transform, position2insert, pose_flatten[element2insert])
            element2insert += 1
        filtered_data = []
        array2transform = array2transform.reshape((6, window))
        array2transform = savgol_filter(array2transform, window, 1, mode='mirror')  # The Savitzky-Golay filter itself
        for row in range(0, 6):
            filtered_data = np.insert(filtered_data, len(filtered_data), array2transform[row, window - 1])
        filtered_data = np.reshape(filtered_data, (-1, 3))
        euler_angle, translation_vec = filtered_data
        euler_angle = np.reshape(euler_angle, (3, 1))
        translation_vec = np.reshape(translation_vec, (3, 1))

        return euler_angle, translation_vec, array2transform


class DrawInfo:
    """
    Draw and show the pose info in the image frame.
    """
    def face_box(self, reprojection_points, landmarks=0):
        """
        Draw the axis box in the detected face.

        :param reprojection_points: array
            The calculated array to project the box vertex.
        :param landmarks: array
            The array that contains the landmarks position to show. landmarks=0 to do not show landmarks.
        :return: The axis box drawn in the image frame.

        """

        line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]  # The box vertex
        for start, end in line_pairs:
            cv2.line(self, reprojection_points[start], reprojection_points[end], (50, 50, 50))
        corner = tuple(reprojection_points[0])
        cv2.line(self, corner, tuple(reprojection_points[1]), (255, 0, 0), 3)
        cv2.line(self, corner, tuple(reprojection_points[3]), (0, 255, 0), 3)
        cv2.line(self, corner, tuple(reprojection_points[4]), (0, 0, 255), 3)
        if landmarks is not 0:  # Draw the landmarks points
            for (x, y) in landmarks:
                cv2.circle(self, (x, y), 1, (220, 220, 220), 1)

    def face_coordinates(self, angle_coord, position_coord):
        """
        Show the face coordinates info.

        :param angle_coord: array
            The orientation coordinates.
        :param position_coord: array
            The position coordinates.
        :return: The face coordinates shown in the image frame.

        """

        cv2.rectangle(self, (10, 10), (200, 100), (255, 255, 255), -1)
        cv2.rectangle(self, (10, 10), (200, 100), (0, 0, 0), 1)
        cv2.putText(self, "* FACE COORDINATES *", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "X: " + "{:7.2f}".format(-10 * position_coord[0, 0]), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "Y: " + "{:7.2f}".format(-10 * position_coord[1, 0]), (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "Z: " + "{:7.2f}".format(-10 * position_coord[2, 0]), (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "a: " + "{:7.2f}".format(angle_coord[0, 0]), (120, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "b: " + "{:7.2f}".format(angle_coord[1, 0]), (120, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "g: " + "{:7.2f}".format(angle_coord[2, 0]), (120, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)

    def probe_arrow(self, reprojection_points, arrow_reproject_points):
        """
        Show the probe indicator arrow.

        :param reprojection_points: array
            The calculated array to project the arrow origin.
        :param arrow_reproject_points: array
            The calculated array to project the arrow indication.
        :return: The probe arrow from the origin to the indication.

        """

        cv2.circle(self, (reprojection_points[:, :, 0], reprojection_points[:, :, 1]), 5, (255, 255, 255), -1)
        cv2.arrowedLine(self, (reprojection_points[:, :, 0], reprojection_points[:, :, 1]),
                        (arrow_reproject_points[:, :, 0], arrow_reproject_points[:, :, 1]),
                        (255, 255, 255), 2, 8, 0, .1)

    def probe_coordinates(self, angle_coord, position_coord):
        """
        Show the probe coordinates info.

        :param angle_coord: array
            The orientation coordinates.
        :param position_coord: array
            The position coordinates.
        :return: The probe coordinates shown in the image frame.

        """

        cv2.rectangle(self, (400, 10), (615, 100), (255, 255, 255), -1)
        cv2.rectangle(self, (400, 10), (615, 100), (0, 0, 0), 1)
        cv2.putText(self, "* PROBE POSE COORDINATES *", (410, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "X: " + "{:7.2f}".format(1000 * position_coord[0, 0]), (420, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "Y: " + "{:7.2f}".format(1000 * position_coord[1, 0]), (420, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "Z: " + "{:7.2f}".format(1000 * position_coord[2, 0]), (420, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "a: " + "{:7.2f}".format(angle_coord[0, 0]), (520, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "b: " + "{:7.2f}".format(angle_coord[1, 0]), (520, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "g: " + "{:7.2f}".format(angle_coord[2, 0]), (520, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)

    def coil_coordinates(self, angle_coord, position_coord):
        """
        Show the coil coordinates info.

        :param angle_coord: array
            The orientation coordinates.
        :param position_coord: array
            The position coordinates.
        :return: The coil coordinates shown in the image frame.

        """

        cv2.rectangle(self, (400, 10), (605, 100), (255, 255, 255), -1)
        cv2.rectangle(self, (400, 10), (605, 100), (0, 0, 0), 1)
        cv2.putText(self, "* COIL POSE COORDINATES *", (410, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "X: " + "{:7.2f}".format(1000 * position_coord[0, 0]), (420, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "Y: " + "{:7.2f}".format(1000 * position_coord[1, 0]), (420, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "Z: " + "{:7.2f}".format(1000 * position_coord[2, 0]), (420, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "a: " + "{:7.2f}".format(angle_coord[0, 0]), (520, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "b: " + "{:7.2f}".format(angle_coord[1, 0]), (520, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
        cv2.putText(self, "g: " + "{:7.2f}".format(angle_coord[2, 0]), (520, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1)
