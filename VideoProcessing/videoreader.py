import numpy as np
import cv2
import matplotlib.pyplot as plt
from hmrconvert import hmrconvert




def main():

    frames_org=get_frames('squat105.mp4')
    # frames_dest=get_frames('squat22.mp4')

    # height, width, layers= np.shape(frames_org[0])
    # # cap_org = cv2.VideoCapture('squat23.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter('output.avi',fourcc, 1,(width,height))
    #
    # ind=0
    #
    # # hmr.initialzation()
    # hmr = hmrconvert()
    # hmr.initialzation()
    # # while ind<len(frames_org) and ind<len(frames_dest):
    # while ind<len(frames_org):
    #     img = hmr.convert3d(frames_org[ind])
    #     img = np.array(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # plt.imshow(img)
    #     # plt.show()
    #     out.write(img)
    #     ind+=1
    #
    # out.release()
    # cv2.destroyAllWindows()


def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    i = 0
    while (cap.isOpened() and i <= 1):
        ret, frame = cap.read()
        print(type(frame))
        height, width, layers = np.shape(frame)
        i += 1
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 1, (width, height))

    frames = []
    i = 0
    hmr = hmrconvert()
    hmr.initialzation()
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % 20 == 0:
            # frames.append(frame)
            img = hmr.convert3d(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out.write(img)
        # print(type(frame))
        # height, width, layers= np.shape(frame)
        # print(height, width)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return frames
#
if __name__ == '__main__':
    main()

# #
# # # cap_org = cv2.VideoCapture('squat23.mp4')
# # #
# # # cap_dest= cv2.VideoCapture('squat22.mp4')
# # # i=0
# # # while(cap_org.isOpened() and i<=1):
# # #     ret, frame = cap_org.read()
# # #     print(type(frame))
# # #     height, width, layers= np.shape(frame)
# # #     print(height, width)
# # #     i+=1
# # # # cap_org = cv2.VideoCapture('squat23.mp4')
# # # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# # # out = cv2.VideoWriter('output.avi',fourcc, 1,(width,height))
# # # i=0
# # # hmr = hmrconvert()
# # # hmr.initialzation()
# # #
# # # while(cap_org.isOpened() and cap_dest.isOpened()):
# #     # ret_org, frame_org = cap_org.read()
# #     # ret_dest, frame_dest = cap_dest.read()
# #     # if ret_org == False or ret_dest == False:
# #     #     break
# #     # if i % 10 == 0: # this is the line I added to make it only save one frame every 20
# #     #     # img= hmr.convert3d(frame_org, frame_dest)
# #     #     img = hmr.convert3d(frame_org)
# #     #     img= np.array(img)
# #     #     img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     #     out.write(img)
# #     #     # cv2.imwrite('kang'+str(i)+'.jpg',frame)
# #     #     # cv2.
# #     #     # cv2.imshow(img, )
# #     #     # plt.imshow(img)
# #     #     # plt.show()
# #     #     # ret=Fals
# #     # i+=1
# # # cap = cv2.VideoCapture('airsquat.mp4')
# # # i=0
# # # while(cap.isOpened() and i<=1):
# # #     ret, frame = cap.read()
# # #     print(type(frame))
# # #     height, width, layers= np.shape(frame)
# # #     i+=1
# # # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# # # out = cv2.VideoWriter('output.avi',fourcc, 1,(width,height))
# # # i=0
# # # hmr = hmrconvert()
# # # hmr.initialzation()
# # # frame1=[]
# # # # while(cap.isOpened()):
# # # #     ret, frame = cap.read()
# # # #     if ret == False:
# # # #         break
# # # #     if i % 20 == 0: # this is the line I added to make it only save one frame every 20
# # # #         img= hmr.convert3d(frame)
# # # #         img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # #         out.write(img)
# # # #         # cv2.imwrite('kang'+str(i)+'.jpg',frame)
# # # #         # cv2.
# # # #         # cv2.imshow(img, )
# # # #         # plt.imshow(img)
# # # #         # plt.show()
# # # #         # ret=Fals
# # # #     i+=1
# # # frame2[]
# # # # #
# #
# # names = ['squat22.mp4', 'squat22.mp4']
# # # window_titles = ['squat23', 'squat24']
# #
# #
# # caps = []
# # for path in names:
# #     cap = cv2.VideoCapture(path)
# #     if not cap.isOpened():
# #         print ("error opening ", path)
# #     else:
# #         caps.append(cap)
# #
# # cap_org = cv2.VideoCapture('squat23.mp4')
# # frame_org=[]
# # i=0
# # while(cap_org .isOpened()):
# #     ret, frame = cap_org .read()
# #     if ret == False:
# #         break
# #     if i % 20 == 0:
# #         frame_org.append(frame)
# #     # print(type(frame))
# #     # height, width, layers= np.shape(frame)
# #     # print(height, width)
# #     i+=1
# # cap_org.release()
# #
# # cap_dest= cv2.VideoCapture('squat23.mp4')
# # frame_dest = []
# # i = 0
# # while (cap_dest.isOpened()):
# #     ret, frame = cap_dest.read()
# #     if ret == False:
# #         break
# #     if i % 20 == 0:
# #         frame_dest.append(frame)
# #         plt.imshow(frame_dest[i])
# #         plt.show()
# #     # print(type(frame))
# #     # height, width, layers= np.shape(frame)
# #     # print(height, width)
# #     i += 1
# # cap_dest.release()
# # height, width, layers= np.shape(frame_org[0])
# # # cap_org = cv2.VideoCapture('squat23.mp4')
# # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# # out = cv2.VideoWriter('output.avi',fourcc, 1,(width,height))
# #
# # ind=0
# # hmr = hmrconvert()
# # hmr.initialzation()
# #
# # while ind<len(frame_org) and ind<len(frame_dest):
# #     img = hmr.convert3d(frame_org[ind], frame_dest[ind])
# #     img = np.array(img)
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     out.write(img)
# #     ind+=1
# #
# # print(len(caps))
# #
# # #
# # #step 2, iterate through videoframes, collect images, and stitch them
# # #
# # # j=0
# # #
# # # while True:
# # #     frames = [] # frames for one timestep
# # #     for cap in caps:
# # #         if cap is not None:
# # #              ret,frame=cap.read()
# # #         if not ret:
# # #             break
# # #         frames.append(frame)
# # #     if len(frames)==2:
# # #         if  j % 10 == 0:
# # #             print(np.shape(frames[0]))
# # #             img = hmr.convert3d(frames[0], frames[1])
# # #             img= np.array(img)
# # #             img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #             out.write(img)
# # #     else:
# # #         break
# # #     j+=1
# #
# #
# #
# # # frames = [None] * len(names);
# # # gray = [None] * len(names);
# # # ret = [None] * len(names);
# # # j=0
# # # hmr = hmrconvert()
# # # hmr.initialzation()
# # # while True:
# # #     for i,c in enumerate(cap):
# # #         if c is not None:
# # #             ret[i], frames[i] = c.read()
# # #     if j % 10 == 0: # this is the line I added to make it only save one frame every 20
# # #         # img= hmr.convert3d(frame_org, frame_dest)
# # #         img = hmr.convert3d(frame[0], frame[1])
# # #         img= np.array(img)
# # #         img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # #         out.write(img)
# # #     j+=1
# # #
# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #        break
# #
# #
# # for cap in caps:
# #     if cap is not None:
# #         cap.release()
# # #
# # # cv2.destroyAllWindows()
# # # cap = cv2.VideoCapture('output.avi')
# # #
# # # while(cap.isOpened()):
# # #     ret, frame = cap.read()
# # #
# # #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #
# # #     cv2.imshow('frame',gray)
# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break
# #
# # # When everything done, release the capture
# # # cap_org.release()
# # # cap_dest.release()
# # # cap.release()
# # cv2.destroyAllWindows()
#
#_____________________________________________________

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from hmrconvert import hmrconvert
#
# cap = cv2.VideoCapture('squat22.mp4')

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


# i = 0
# while (cap.isOpened() and i <= 1):
#     ret, frame = cap.read()
#     print(type(frame))
#     height, width, layers = np.shape(frame)
#     i += 1
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', fourcc, 1, (width, height))
# i = 0
# hmr = hmrconvert()
# hmr.initialzation()
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     if i % 20 == 0:  # this is the line I added to make it only save one frame every 20
#         img = hmr.convert3d(frame)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         out.write(img)
#         # cv2.imwrite('kang'+str(i)+'.jpg',frame)
#         # cv2.
#         # cv2.imshow(img, )
#         # plt.imshow(img)
#         # plt.show()
#         # ret=Fals
#     i += 1
#
# #
# # cap = cv2.VideoCapture('output.avi')
# #
# # while(cap.isOpened()):
# #     ret, frame = cap.read()
# #
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #
# #     cv2.imshow('frame',gray)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
