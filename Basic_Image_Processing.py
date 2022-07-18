
from tracemalloc import start
from turtle import end_fill
import cv2
import numpy as np

# Loading an image
img_path = "D:\\OpenCV\\images\\space.jpg"
image = cv2.imread(img_path)
# print(image)

# Displaying image
cv2.imshow('sample_image', image)

cv2.waitKey(0)     # wait until i press any key
cv2.destroyAllWindows()  # destroy all windows


# Saving an image
filename = "SavedTestimage.jpg"
cv2.imwrite(filename, image)
print("Image Saved Successfully!!")


# Accessing Image Properties
print(image.shape)

print("total Color Channels: %d" % image.shape[2])

# Changing Color-Space
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('original_image', image)
cv2.imshow('gray_sample_image', gray_image)

cv2.waitKey(0)     # wait until i press any key
cv2.destroyAllWindows()  # destroy all windows

# Resizing the image
resized_image = cv2.resize(image, (800,800))
cv2.imshow('original_image', image)
cv2.imshow('resized_image', resized_image)

cv2.waitKey(0)     # wait until i press any key
cv2.destroyAllWindows()  # destroy all windows

# Displaying text

text="This image is taken by NASA"
coordinate=(100,200)
font=cv2.FONT_HERSHEY_SIMPLEX
fontscale=1
color=(255,0,0)
thickness=2

image_with_text = cv2.putText(image,text, coordinate, font, fontscale, color, thickness)
cv2.imshow('image_with_text', image_with_text)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Drawing a line
coordinate=(100,200)
color=(255,255,255)
thickness=2
start_point=(0,0)
end_point=(250,250)

image_with_line = cv2.line(image, start_point, end_point, color, thickness)

cv2.imshow('image_with_line', image_with_line)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Displaying a circle
center_coordiates=(250,250)
color=(255,0,255)
thickness=2
radius=20

image_with_circle = cv2.circle(image, center_coordiates, radius, color, thickness)

cv2.imshow('image_with_circle', image_with_circle)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Displaying a Rectangle
start_point=(120,50)
end_point=(250,250)
color=(0,0,255)
thickness=2

image_with_rectangle = cv2.rectangle(image,start_point, end_point, color, thickness)

cv2.imshow('image_with_rectangle', image_with_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Displaying an ellipse
color=(0,0,255)
thickness=2
center_coordinate=(120,100)
axeslength=(100,50)
angle=30
startangle=0
endangle=360


image_with_ellipse = cv2.ellipse(image, center_coordinate, axeslength, angle, startangle, endangle,
                                 color, thickness)

cv2.imshow('image_with_ellipse', image_with_ellipse)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Display Images in multipule modes

# 1 for color image
# 0 for gray image
# -1 for no change

cv2.imshow("originalimage", image)

# gray image
gray_image = cv2.imread(img_path,0)
cv2.imshow("gray_image", gray_image)

#colored image
colored_image = cv2.imread(img_path,1)
cv2.imshow("colored_image", colored_image)

#Nochange image
noChange_image = cv2.imread(img_path,-1)
cv2.imshow("noChange_image", noChange_image)


cv2.waitKey(0)
cv2.destroyAllWindows()