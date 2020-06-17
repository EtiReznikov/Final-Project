# @inProceedings{kanazawaHMR18,
#   title={End-to-end Recovery of Human Shape and Pose},
#   author = {Angjoo Kanazawa
#   and Michael J. Black
#   and David W. Jacobs
#   and Jitendra Malik},
#   booktitle={Computer Vision and Pattern Recognition (CVPR)},
#   year={2018}
# }
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from tkinter import *
# from tkinter import messagebox
from absl import flags
import numpy as np
#
from tkinter import filedialog
from tkinter import *
import skimage.io as io
import tensorflow as tf
from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel
from PIL import Image, ImageTk
import math
import cv2
import matplotlib.pyplot as plt
class hmr2convert:


    def __init__(self, master=None):
        self.config = flags.FLAGS
        self.config(sys.argv)
        # Using pre-trained model, change this to use your own.
        self.config.load_path = src.config.PRETRAINED_MODEL

        self.config.batch_size = 1

        self.renderer = vis_util.SMPLRenderer(face_path=self.config.smpl_face_path)


#
    def visualize(self, img_org, img_dest, proc_param_org, proc_param_dest, joints_org, joints_dest, verts_org,
                  verts_dest, cam_org, cam_dest):

        cam_for_render_org, vert_shifted_org, joints_orig_org = vis_util.get_original(
            proc_param_org, verts_org, cam_org, joints_org, img_size=img_org.shape[:2])
        #
        cam_for_render_dest, vert_shifted_dest, joints_orig_dest = vis_util.get_original(
            proc_param_dest, verts_dest, cam_org, joints_dest, img_size=img_dest.shape[:2])

        rend_img_overlay_org = self.renderer(
            vert_shifted_org, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
            near=None,
            color_id=1,
            img_size=img_org.shape[:2])

        rend_img_overlay_dest = self.renderer(
            vert_shifted_dest, cam=cam_for_render_dest, img=None, do_alpha=True, far=None,
            near=None,
            color_id=0,
            img_size=img_dest.shape[:2])

        Image3d_org = Image.fromarray(rend_img_overlay_org, mode='RGBA')
        Image3d_dest = Image.fromarray(rend_img_overlay_dest, mode='RGBA')

        mask = Image.new("L", Image3d_dest.size, 128)
        im = Image.composite(Image3d_dest, Image3d_org, mask)
        #
        #
        # plt.imshow(im)
        # plt.show()
        #

        # fig, ax = plt.subplots(1)
        # ax.imshow(im)

        return im

    def preprocess_image(self, img):
        # img = io.imread(img_path)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # if json_path is None:
        if np.max(img.shape[:2]) != self.config.img_size:
            print('Resizing so the max image size is %d..' % self.config.img_size)
            scale = (float(self.config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
        # else:
        #     scale, center = op_util.get_bbox(json_path)

        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                   self.config.img_size)

        # Normalize image to [-1, 1]dest [[ 81.447205 202.8527  ]]


        crop = 2 * ((crop / 255.) - 0.5)

        return crop, proc_param, img

#     # def initialzation(self, img_path, json_path=None):
#     # def initialzation( self, img):
    def initialzation(self):
        sess = tf.Session()
        self.model = RunModel(self.config, sess=sess)
#
#
#     # #
    def convert3d(self,img_org, img_dest):
        # input_img, proc_param, img = self.preprocess_image( img_path, json_path)
        input_img_org, proc_param_org, img1= self.preprocess_image( img_org)

        # Add batch dimension: 1 x D x D x 3
        input_img_org = np.expand_dims(input_img_org, 0)



        input_img_dest, proc_param_dest, img2 = self.preprocess_image(img_dest)
        # Add batch dimension: 1 x D x D x 3
        input_img_dest = np.expand_dims(input_img_dest, 0)

        joints_org, verts_org, cams_org, joints3d_org, theta_org = self.model.predict(
            input_img_org, get_theta=True)

        joints_dest, verts_dest, cams_dest, joints3d_dest, theta_dest = self.model.predict(
            input_img_dest, get_theta=True)


        print(np.shape(verts_dest))

        dist= verts_org-verts_dest
        # verts_dest+=dist
        dist=get_ankles_dist(joints_org, joints_dest)
        joints_dest[0]+=dist

        # joints_dest=joints_org
        # verts_dest=verts_org
        # if (joints_dest[:,0][0][0]==joints_org[:,0][0][0] and joints_dest[:,0][0][1]==joints_org[:,0][0][1] ):
        #     print("true")
        # else:
        #     print("false")

        # print(np.shape(joints_dest))
        # print(np.shape(verts_dest))
        # print(np.shape(cams_dest))

        # #
        #
        height_dest = get_hegiht(joints_dest)
        height_org = get_hegiht(joints_org)
        print(height_dest)
        print(height_org)
        # #
        # #
        scale = height_org / height_dest
        joints_dest *= scale
        height_dest = get_hegiht(joints_dest)
        #
        # print(joints_dest)
        # # #
        #
        #
        # proc_param_dest=proc_param_org
        # print(joints_dest)
        # print("dest", joints_dest[:,0])

        # print(type(proc_param_dest))
        # print(proc_param_dest)
        # print(proc_param_org)
        # #
        # img2*=scale
        # proc_param_dest*=scale
        # joints_dest*=scale
        # # verts_dest*=scale
        # cams_dest*=scale


        return self.visualize( img1, img2, proc_param_org, proc_param_dest, joints_org[0], joints_dest[0],
                               verts_org[0], verts_dest[0], cams_org[0], cams_dest[0])


    def get_joints3d(self,img):
        # input_img, proc_param, img = self.preprocess_image( img_path, json_path)
        input_img, proc_param, img = self.preprocess_image( img)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
        # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        joints, verts, cams, joints3d, theta = self.model.predict(
            input_img, get_theta=True)

        joints3d=np.reshape(joints3d, (19,3))
        return joints3d


    def get_parmeters(self,img):
        # input_img, proc_param, img = self.preprocess_image( img_path, json_path)
        input_img, proc_param, img = self.preprocess_image( img)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
        # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        joints, verts, cams, joints3d, theta = self.model.predict(
            input_img, get_theta=True)

        return joints, verts, cams, joints3d, theta,  input_img, proc_param, img

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

def get_ankles_dist(joints_src, joints_dest):
    right_ankle_org= joints_src[:,0]
    right_ankle_dest= joints_dest[:,0]
    right_ankle_org=right_ankle_org[0]
    right_ankle_dest= right_ankle_dest[0]

    dist= right_ankle_org-right_ankle_dest

    return dist



# def main ():
#     hmr= hmr2convert()
#     hmr.initialzation()
#     img= cv2.imread("A31.jpg")
#     img2= cv2.imread("A3.jpeg")
#     hmr.convert3d(img2, img)
#
# if __name__:
#     main()