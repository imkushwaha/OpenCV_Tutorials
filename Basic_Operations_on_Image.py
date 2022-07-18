import numpy as np
import cv2

img_path = "C:\\OpenCV\\images\\space.jpg"
img = cv2.imread(img_path)
image = cv2.resize(img, (600, 600))


# Acess Pixel values and modeify them
px = img[100,100]  # a particular pixel
print(px)

blue_channel = img[100, 100, 0]   # can also do like: img[100, 100][0]
print(blue_channel)

img[100, 100] = [255, 255, 255]
print(img[100, 100])

# Access Image Properties

img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # read the image in color scheme
alpha_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # give alpha image
gray_Img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # give gray image

print('RGB Shape: ', img.shape)
print('ARGB Shape: ', alpha_img.shape)
print('Gray Shape: ', gray_Img.shape)

# data type
print("image datatype: ", img.dtype)

# size of image
print('image size: ', img.size)


# Setting Region of Image

img_raw = cv2.imread(img_path)

roi = cv2.selectROI(img_raw)
print(roi)

# cropping selected ROI from the raw(given) image

roi_cropped = img_raw[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])]
cv2.imshow("ROI image", roi_cropped)
cv2.imwrite("cropped.jpeg", roi_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Splitting and merging images
img_path = "C:\\OpenCV\\images\\space.jpg"
img = cv2.imread(img_path)
image = cv2.resize(img, (600, 600))

g,b,r = cv2.split(image)
cv2.imshow("Green Scale of the image", g)
cv2.imshow("Blue Scale of the image", b)
cv2.imshow("Red Scale of the image", r)

img1 = cv2.merge((g,b,r))
cv2.imshow("image after merger of three channel", img1)


cv2.waitKey(0)
cv2.destroyAllWindows()

# Change the image color

img = cv2.imread(img_path)
image = cv2.resize(img, (600, 600))

color_changed = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
cv2.imshow("Changed color scheme", color_changed)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Blend two different images: Mixing two images, one should be base and second image is for additive purpose

src1 = cv2.imread("C:\\OpenCV\\images\\space.jpg", cv2.IMREAD_COLOR)
src2 = cv2.imread("C:\\OpenCV\\images\\test.jpg", cv2.IMREAD_COLOR)

img1 = cv2.resize(src1, (800,600))
img2 = cv2.resize(src2, (800,600))

blended_image = cv2.addWeighted(img1, 1, img2, 0.5, 0.0)
# 1 is alpha value for first image intensity
# 0.5 is beta value for second or additive image
# 0.0 is gama

cv2.imshow("Blended/Additive image", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply different filters on Image

img = cv2.imread("C:\\OpenCV\\images\\test.jpg")
k_sharped = np.array([[-1, -1, -1],
                      [-1, 9, 1],
                      [-1, -1, -1]])

sharpened_image = cv2.filter2D(img, -1, k_sharped)  # -1 is for depth

cv2.imshow("Original image", img)
cv2.imshow("Filtered image", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image Thresholding

img = cv2.imread("C:\\OpenCV\\images\\space.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(img, (600, 600))
ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cannyImage = cv2.Canny(image,50,100)  # canny is edge detector technique used widely

cv2.imshow("Original image", image)
cv2.imshow("Thresholded image", thresh)
cv2.imshow("Canny image", cannyImage)


cv2.waitKey(0)
cv2.destroyAllWindows()

# Contour Detection and Shape Detection

import matplotlib.pyplot as plt

img = cv2.imread("C:\\OpenCV\\images\\shape.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SETTING THRESHOLD OF THE GRAY SCALE IMAGE to convert image into binary
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# CONTOURS USING FINDCONTOURS FUNCTION

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

i = 0

for contour in contours:
    if i == 0:
        i = 1
        continue
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [contour], 0, (255, 0, 255), 5)
    # finding the center of different shapes
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

    # i want to put name of the shape inside the corresponding shapes

    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    elif len(approx) == 4:
        cv2.putText(img, "Quadrilateral", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif len(approx) == 6:
        cv2.putText(img, "Hexagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Displaying things

cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Color Detection

img = cv2.imread("C:\\OpenCV\\images\\shape.jpg")

# HSV: Hue, Saturation and value. HSV is common used in color and paint softwares

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower_blue = np.array([0, 50, 50])
# upper_blue = np.array([140, 255, 255])

lower_yellow = np.array([10, 100, 20])
upper_yellow = np.array([25, 255, 255])

# threshold the HSV image to get only blue colors

# mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

res = cv2.bitwise_and(img, img, mask=mask_yellow)
cv2.imshow("res", res)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Object replacing in 2D image using OpenCV

from matplotlib import pyplot as plt

img = cv2.imread('C:\\OpenCV\\images\\space.jpg', cv2.IMREAD_COLOR)

img1 = img.copy()
mask = np.zeros((100, 300, 3))
print(mask.shape)

pos = (200, 200)
var = img1[200:(200+mask.shape[0]), 200:(200+mask.shape[1])] = mask

cv2.imshow("coloring", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()






