import sys

if sys.version_info[0] < 3:
    raise Exception("Ërror: Python 3 or a more recent version is required.")

import cv2, time

streamer = cv2.VideoCapture(0)
url = "http://192.168.225.26:8080/video"
streamer.open(url)

while True:
    ( _, frame) = streamer.read()
    cv2.imshow("Press key \"q\" to exit...", frame)
    key = cv2.waitKey(1)

    if key==ord('q'):
        break

streamer.release()
cv2.destroyAllWindows()
