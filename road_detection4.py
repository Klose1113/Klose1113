import matplotlib.pylab as plt
import cv2
import numpy as np

low_green = np.array([25, 52, 70])
high_green = np.array([102, 255, 255])

low_destroy = np.array([18,0,140])
high_destroy = np.array([179,255,255])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    #left_line = make_coordinates(image, left_fit_average)
    #right_line = make_coordinates(image, right_fit_average)
    #return np.array([left_line, right_line])


def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

img = cv2.imread('road8.jpg')

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(imgHSV, low_destroy, high_destroy)

#mask = 255-mask

mask_2 = cv2.inRange(imgHSV, low_green, high_green)

mask_2 = 255-mask_2

res = cv2.bitwise_and(img, img, mask = mask)

res_2 = cv2.bitwise_and(res, res, mask = mask_2)

def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_image = cv2.GaussianBlur(gray_image, (7,7),0)
    canny_image = cv2.Canny(gaussian_image, 50, 200)

    lines = cv2.HoughLinesP(canny_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=150,
                            lines=np.array([]),
                            minLineLength=100,
                            maxLineGap=40)

    image_with_lines = draw_the_lines(img, lines)
    return image_with_lines


plt.subplot(1,2,1),plt.imshow(process(res_2))
plt.subplot(1,2,2),plt.imshow(img)


plt.show()
