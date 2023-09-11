
import matplotlib.pylab as plt
import cv2
import numpy as np


def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

imag = cv2.imread('road1.jpg')
image = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)

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
    gaussian_image = cv2.GaussianBlur(gray_image, (13,13),0)
    canny_image = cv2.Canny(gaussian_image, 50, 200)

    lines = cv2.HoughLinesP(canny_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=100,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(gray, (13,13),0)

plt.subplot(2,2,1),plt.imshow(process(image))
plt.subplot(2,2,2),plt.imshow(image)
plt.subplot(2,2,3),plt.imshow(gray)
plt.subplot(2,2,4),plt.imshow(gaussian)


plt.show()
