import cv2
from PIL import Image
from util import get_limits

# Define colors in BGR
colors = {
    "Red": [0, 0, 255],
    "Yellow": [0, 255, 255],
    "Black": [0, 0, 0]
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, bgr_value in colors.items():
        lowerLimit, upperLimit = get_limits(color=bgr_value)
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Put color name text
            cv2.putText(frame, color_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Color Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

   
