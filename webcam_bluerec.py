# Python code for Multiple Color Detection


import numpy as np
import cv2


# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while(1):

	_, frame = webcam.read()


	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


	blue_lower = np.array([110, 50, 50], np.uint8)
	blue_upper = np.array([130, 255, 255], np.uint8)
	mask = cv2.inRange(hsv, blue_lower, blue_upper)


	kernel = np.ones((5, 5), "uint8")


	blue_mask = cv2.dilate(blue_mask, kernel)

	res_blue = cv2.bitwise_and(frame, frame,
							mask = blue_mask)



	contours, hierarchy = cv2.findContours(blue_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 300):
			x, y, w, h = cv2.boundingRect(contour)
			frame = cv2.rectangle(frame, (x, y),
									(x + w, y + h),
									(255, 0, 0), 2)

			cv2.putText(frame, "Blue Colour", (x, y),
						cv2.FONT_HERSHEY_SIMPLEX,
						1.0, (255, 0, 0))


	cv2.imshow("Multiple Color Detection in Real-TIme", res_blue)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		webcam.release()
		cv2.destroyAllWindows()
		break

