import cv2

from detector.hand_detector import HandDetector
from utils.fps import FPS

cap = cv2.VideoCapture(0)

detector = HandDetector()

fpsCounter = FPS()

while True:

    success, img = cap.read()

    if not success:
        break

    img = detector.findHands(img)

    lmList = detector.findPosition(img)

    if lmList:

        x, y = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

    fps = fpsCounter.update()

    cv2.putText(
        img,
        f"FPS : {fps}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("AI Hand Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()