import cv2
import numpy as np
cap = cv2.VideoCapture('/home/jaggi/competition_videos/verygood_describe/prievewout_video_20201031-143221.avi')

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # preparing the mask to overlay
    # Threshold of blue in HSV space
    lower_blue = np.array([0, 250, 0])
    upper_blue = np.array([0, 255, 0])
    mask = cv2.inRange(frame, lower_blue, upper_blue)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    # Display the resulting frame

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
