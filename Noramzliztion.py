import cv2
import numpy as np
import math
from hmr2convert import hmr2convert

import matplotlib.pyplot as plt



def get_hegiht(joints):
    right_ankle= joints[:,0]
    left_ankle= joints[:,5]
    right_ankle_y=right_ankle[0][1]
    left_ankle_y=left_ankle[0][1]
    if (right_ankle_y< left_ankle_y):
        ankle=right_ankle[0]
    else:
        ankle=left_ankle[0]

    neck= joints[:,12][0]

    height= math.sqrt((neck[1]-ankle[1])**2+(neck[0]-ankle[0])**2)
    return height


img_org = cv2.imread("A3.jpeg")
img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

img_dest = cv2.imread("short_man_6.jpg")
img_dest = cv2.cvtColor(img_dest, cv2.COLOR_BGR2RGB)

hmr = hmr2convert()
hmr.initialzation()
result_regular= hmr.convert3d(img_org,img_dest)
joints_org, verts_org, cams_org, joints3d_org, theta_org, input_img_org, proc_param_org, img_org=  hmr.get_parmeters(img_org)
joints_dest, verts_dest, cams_dest, joints3d_dest, theta_dest, input_dest, proc_param_dest, img_dest= hmr.get_parmeters(img_dest)

height_dest= get_hegiht(joints_dest)
height_org= get_hegiht(joints_org)
print("dest", height_dest)
print("org", height_org)
scale=  height_org /height_dest
print(scale)
# print(joints_dest)
# print(joints_org)

joints_dest= joints_dest*scale
joints3d_dest= joints3d_dest*scale
verts_dest= verts_dest*scale

height_dest=get_hegiht(joints_dest)
print("dest", height_dest)
# print(joints_dest)
# print(joints_org)

result= hmr.visualize( img_org, img_dest, proc_param_org, proc_param_dest, joints_org[0], joints_dest[0],
                               verts_org[0], verts_dest[0], cams_org[0], cams_dest[0])
plt.figure()
plt.imshow(result_regular)
plt.title("without normal")
plt.figure()
plt.title("with normal")
plt.imshow(result)
plt.show()




# img=  hmr.visualize( img_t, imgs, proc_param_org, proc_param_dest, joints_org[0], joints_dest[0],
#                                verts_org[0], verts_dest[0], cams_org[0], cams_dest[0])
#






# joints:
#    0: Right ankle
#    1: Right knee
#    2: Right hip
#    3: Left hip
#    4: Left knee
#    5: Left ankle
#    6: Right wrist
#    7: Right elbow
#    8: Right shoulder
#    9: Left shoulder
#    10: Left elbow
#    11: Left wrist
#    12: Neck
#    13: Head top
#    14: nose
#    15: left_eye
#    16: right_eye
#    17: left_ear
#    18: right_ear
#    """
