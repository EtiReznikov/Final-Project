
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

class hmr_rendrer :

    def __init__(self, master=None):
        self.config = flags.FLAGS
        self.config(sys.argv)
        # Using pre-trained model, change this to use your own.
        self.config.load_path = src.config.PRETRAINED_MODEL

        self.config.batch_size = 1

        self.renderer = vis_util.SMPLRenderer(face_path=self.config.smpl_face_path)
        # Frame.__init__(self, master)
        # self.master = master
        # self.init_window()

    #
    # def get_render(self, img, proc_param, joints, verts, cam):
    #
    #
    #     cam_for_render, vert_shifted, jointss_orig= vis_util.get_original(
    #         proc_param, verts, cam, joints, img_size=img.shape[:2])
    #
    #     print(np.shape(joints))
    #
    #     rend_img_overlay = self.renderer(
    #         vert_shifted, cam=cam_for_render, img=None, do_alpha=True, far=None,
    #     # near=None,
    #     color_id=0,
    #     img_size=img.shape[:2])
    #
    #     print(type(rend_img_overlay[1]))
    #     print(np.shape(rend_img_overlay[1]))
    #
    #     # image3d= Image.fromarray(rend_img_overlay)
    #     # image3d = image3d.resize((224, 224), Image.ANTIALIAS)
    #     # render = ImageTk.PhotoImage(image3d)
    #
    #     return joints
    #     # otherimg= Label(self, image=render)
    #     # otherimg.image=render
    #     # otherimg.place(x=1, y=1)

    # def preprocess_image(self,  img_path, json_path=None):
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

        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        return crop, proc_param, img

    # def initialzation(self, img_path, json_path=None):
    # def initialzation( self, img):
    def initialzation(self):
        sess = tf.Session()
        self.model = RunModel(self.config, sess=sess)



    def convert3d(self,img):
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
