import cv2
import dlib
from imutils import face_utils


# Determine the predictor configuration
face_landmark_path = r'Predictors\shape_predictor_5_face_landmarks.dat'
# face_landmark_path = r'Predictors\shape_predictor_68_face_landmarks.dat'
# face_landmark_path = r'Predictors\shape_predictor_81_face_landmarks.dat'
# face_landmark_path = r'Predictors\shape_predictor_194_face_landmarks.dat'


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to connect to camera.")
        return

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    while cap.isOpened():
        frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Turns the operating frame in gray
        face_rects = detector(gray_frame, 0)  # Detect the face in the image frame

        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:  # Draw the predicted face points
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        cv2.imshow("Face Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break


if __name__ == '__main__':
    main()
