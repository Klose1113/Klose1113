import cv2
import numpy as np
def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 80, 150)

    height, width = edges.shape
    roi_vertices = [(0, height*0.6), (width //2 , height*0.25), (width, height*0.6)]

    mask = np.zeros_like(edges)

    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)

    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=25, minLineLength=10, maxLineGap=80)

    line_image = np.copy(image) * 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)

    result = cv2.addWeighted(image, 1, line_image, 1, 0)

    return result

cap = cv2.VideoCapture("test1.mp4")
while True:
    ret, frame = cap.read()

    if not ret:
        break

    result_image = detect_lanes(frame)

    cv2.imshow('Lane Detection', result_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
