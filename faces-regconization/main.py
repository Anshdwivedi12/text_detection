import cv2
import mediapipe as mp

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame color to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
                w_box, h_box = int(bboxC.width * w), int(bboxC.height * h)

                # Crop and blur the face
                face = frame[y:y+h_box, x:x+w_box]
                face = cv2.GaussianBlur(face, (99, 99), 30)
                frame[y:y+h_box, x:x+w_box] = face

        cv2.imshow("Face Anonymizer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
