import numpy as np
import cv2
import matplotlib.pyplot as plt
from hmrconvert import hmrconvert
cap = cv2.VideoCapture('airsquat.mp4')

#
# img1 = cv2.imread('squat1.jpg')
# img2 = cv2.imread('squat2.jpg')
# img3 = cv2.imread('Mysquat.jpg')
#
# height , width , layers =  np.shape(img1)
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# video = cv2.VideoWriter('video.avi',fourcc,1,(width,height))
#
# video.write(img1)
# video.write(img2)
# video.write(img3)
#
# cv2.destroyAllWindows()
# video.release()

# cap = cv2.VideoCapture(0)
#
# # Define the codec and create VideoWriter object
i=0
while(cap.isOpened() and i<=1):
    ret, frame = cap.read()
    print(type(frame))
    height, width, layers= np.shape(frame)
    i+=1
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 1,(width,height))
i=0
hmr = hmrconvert()
hmr.initialzation()
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i % 20 == 0: # this is the line I added to make it only save one frame every 20
        img= hmr.convert3d(frame)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(img)
        # cv2.imwrite('kang'+str(i)+'.jpg',frame)
        # cv2.
        # cv2.imshow(img, )
        # plt.imshow(img)
        # plt.show()
        # ret=Fals
    i+=1
    

#
# cap = cv2.VideoCapture('output.avi')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

