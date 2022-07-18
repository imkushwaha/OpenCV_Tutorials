
# importing cv2

import cv2

# Open Camera using OpenCv

cap = cv2.VideoCapture(0)   # will start our system inbuilt camera
while(True):
    ret_, frame = cap.read()
    cv2.imshow("sampleVideo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):    # press q button or key to close the loop of frame i.e.. video
        break
        cap.release()   # release the space taken by video

cv2.destroyAllWindows()

# Capture video using OpenCV and save it in directory

cap = cv2.VideoCapture(0)

if(cap.isOpened() == False):
    print("Camera could not open")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# video coded ... encoders and decoders
video_cod = cv2.VideoWriter_fourcc(*'XVID')   # video coded used for encoding and decoding video
video_output = cv2.VideoWriter('Captured_Video.Mp4', video_cod, 30, (frame_width, frame_height))#  30 is FPS

while(True):
    ret, frame = cap.read()

    if ret==True:
        video_output.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
video_output.release()
cv2.destroyAllWindows()

print("The video was saved successfully")


# Playing video from file

cap = cv2.VideoCapture("Captured_Video.Mp4")

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


