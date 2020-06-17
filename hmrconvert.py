# # @inProceedings{kanazawaHMR18,
# #   title={End-to-end Recovery of Human Shape and Pose},
# #   author = {Angjoo Kanazawa
# #   and Michael J. Black
# #   and David W. Jacobs
# #   and Jitendra Malik},
# #   booktitle={Computer Vision and Pattern Recognition (CVPR)},
# #   year={2018}
# # }
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# # from tkinter import *
# # from tkinter import messagebox
# from absl import flags
# import numpy as np
# #
# from tkinter import filedialog
# from tkinter import *
# import skimage.io as io
# import tensorflow as tf
# from src.util import renderer as vis_util
# from src.util import image as img_util
# from src.util import openpose as op_util
# import src.config
# from src.RunModel import RunModel
# from PIL import Image, ImageTk
# import cv2
# import matplotlib.pyplot as plt
# class hmrconvert:
#
#
#     def __init__(self, master=None):
#         self.config = flags.FLAGS
#         self.config(sys.argv)
#         # Using pre-trained model, change this to use your own.
#         self.config.load_path = src.config.PRETRAINED_MODEL
#
#         self.config.batch_size = 1
#
#         self.renderer = vis_util.SMPLRenderer(face_path=self.config.smpl_face_path)
#
#     # def visualize(self, img_org, img_dest, proc_param_org, proc_param_dest, joints_org, joints_dest, verts_org, verts_dest, cam_org, cam_dest):
#     #
#     #     cam_for_render_org, vert_shifted_org, joints_orig_org = vis_util.get_original(
#     #         proc_param_org, verts_org, cam_org, joints_org, img_size=img_org.shape[:2])
#     #
#     #     # cam_for_render_dest, vert_shifted_dest, joints_orig_dest = vis_util.get_original(
#     #     #     proc_param_dest, verts_dest, cam_org, joints_dest, img_size=img_org.shape[:2])
#     #
#     #
#     #
#     #     rend_img_overlay_org = self.renderer(
#     #         vert_shifted_org, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #              near=None,
#     #              color_id=1,
#     #              img_size=img_org.shape[:2])
#     #
#
#     def visualize(self, img, proc_param, joints, verts, cam):
#         cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
#             proc_param, verts, cam, joints, img_size=img.shape[:2])
#         rend_img_overlay = self.renderer(
#             vert_shifted, cam=cam_for_render, img=None, do_alpha=True, far=None,
#             near=None,
#             color_id=0,
#             img_size=img.shape[:2])
#
#         # image3d= Image.fromarray(rend_img_overlay)
#         # image3d = image3d.resize((224, 224), Image.ANTIALIAS)
#         # render = ImageTk.PhotoImage(image3d)
#
#         return rend_img_overlay
#
#         # rend_img_vp1_org = self.renderer.rotated(
#         #     vert_shifted_org, 60, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#         #          near=None,
#         #          color_id=1,
#         #          img_size=img_org.shape[:2])
#         #
#         # rend_img_vp2_org = self.renderer.rotated(
#         #     vert_shifted_org, -60, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#         #          near=None,
#         #          color_id=1,
#         #          img_size=img_org.shape[:2])
#         #
#         # rend_img_overlay_dest = self.renderer(
#         #     vert_shifted_dest, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#         #          near=None,
#         #          color_id=0,
#         #          img_size=img_org.shape[:2])
#         # rend_img_vp1_dest = self.renderer.rotated(
#         #     vert_shifted_dest, 60, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#         #     near=None,
#         #     color_id=0,
#         #     img_size=img_org.shape[:2])
#         #
#         # rend_img_vp2_dest = self.renderer.rotated(
#         #     vert_shifted_dest, -60, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#         #     near=None,
#         #     color_id=0,
#         #     img_size=img_org.shape[:2])
#
#
#         #
#         #
#         # Image3d_org = Image.fromarray(rend_img_overlay_org, mode='RGBA')
#         # # Image3d_dest = Image.fromarray(rend_img_overlay_dest)
#         # # Image3d_org.paste( Image3d_dest, (0, 0),  Image3d_dest)
#         # # print(np.dtype(Image3d_org))
#         # return Image3d_org
#         #
#
#     # def visualize(self, img_org, img_dest, proc_param_org, proc_param_dest, joints_org, joints_dest, verts_org,
#     #               verts_dest, cam_org, cam_dest):
#     #
#     #     cam_for_render_org, vert_shifted_org, joints_orig_org = vis_util.get_original(
#     #         proc_param_org, verts_org, cam_org, joints_org, img_size=img_org.shape[:2])
#     #
#     #     cam_for_render_dest, vert_shifted_dest, joints_orig_dest = vis_util.get_original(
#     #         proc_param_dest, verts_dest, cam_org, joints_dest, img_size=img_org.shape[:2])
#     #
#     #     rend_img_overlay_org = self.renderer(
#     #         vert_shifted_org, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #         near=None,
#     #         color_id=1,
#     #         img_size=img_org.shape[:2])
#     #
#     #     rend_img_vp1_org = self.renderer.rotated(
#     #         vert_shifted_org, 60, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #         near=None,
#     #         color_id=1,
#     #         img_size=img_org.shape[:2])
#     #
#     #     rend_img_vp2_org = self.renderer.rotated(
#     #         vert_shifted_org, -60, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #         near=None,
#     #         color_id=1,
#     #         img_size=img_org.shape[:2])
#     #
#     #     rend_img_overlay_dest = self.renderer(
#     #         vert_shifted_dest, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #         near=None,
#     #         color_id=0,
#     #         img_size=img_org.shape[:2])
#     #
#     #     rend_img_vp1_dest = self.renderer.rotated(
#     #         vert_shifted_dest, 60, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #         near=None,
#     #         color_id=0,
#     #         img_size=img_org.shape[:2])
#     #
#     #     rend_img_vp2_dest = self.renderer.rotated(
#     #         vert_shifted_dest, -60, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #         near=None,
#     #         color_id=0,
#     #         img_size=img_org.shape[:2])
#     #
#     #     Image3d_org = Image.fromarray(rend_img_overlay_org, mode='RGBA')
#     #     Image3d_dest = Image.fromarray(rend_img_overlay_dest, mode='RGBA')
#     #
#     #     mask = Image.new("L", Image3d_dest.size, 128)
#     #     im = Image.composite(Image3d_dest, Image3d_org, mask)
#     #     # fig, ax = plt.subplots(1)
#     #     # ax.imshow(im)
#     #
#     #     return im
#     # #
#     # def visualize(self, img_org, img_dest, proc_param_org, proc_param_dest, joints_org, joints_dest, verts_org,verts_dest, cam_org, can_dest):
#     #     cam_for_render_org, vert_shifted_org, joints_orig_org = vis_util.get_original(
#     #         proc_param_org, verts_org, cam_org, joints_org, img_size=img_org.shape[:2])
#     #
#     #     cam_for_render_dest, vert_shifted_dest, joints_orig_dest = vis_util.get_original(
#     #         proc_param_dest, verts_dest, cam_org, joints_dest, img_size=img_org.shape[:2])
#     #
#     #     rend_img_overlay_org = self.renderer(
#     #         vert_shifted_org, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #         near=None,
#     #         color_id=1,
#     #         img_size=img_org.shape[:2])
#     #
#     #     rend_img_overlay_dest = self.renderer(
#     #         vert_shifted_dest, cam=cam_for_render_org, img=None, do_alpha=True, far=None,
#     #         near=None,
#     #         color_id=0,
#     #         img_size=img_org.shape[:2])
#     #
#     #     Image3d_org = Image.fromarray(rend_img_overlay_org, mode='RGBA')
#     #     Image3d_dest = Image.fromarray(rend_img_overlay_dest)
#     #
#     #     mask = Image.new("L", Image3d_dest.size, 128)
#     #     im = Image.composite(Image3d_dest, Image3d_org, mask)
#     #     # Image3d_org.paste( Image3d_dest, (0, 0),  Image3d_dest)
#     #     # print(np.dtype(Image3d_org))
#     #     return im
#
#     # def preprocess_image(self,  img_path, json_path=None):
#     def preprocess_image(self, img):
#         # img = io.imread(img_path)
#         if img.shape[2] == 4:
#             img = img[:, :, :3]
#
#         # if json_path is None:
#         if np.max(img.shape[:2]) != self.config.img_size:
#             print('Resizing so the max image size is %d..' % self.config.img_size)
#             scale = (float(self.config.img_size) / np.max(img.shape[:2]))
#         else:
#             scale = 1.
#         center = np.round(np.array(img.shape[:2]) / 2).astype(int)
#         # image center in (x,y)
#         center = center[::-1]
#         # else:
#         #     scale, center = op_util.get_bbox(json_path)
#
#         crop, proc_param = img_util.scale_and_crop(img, scale, center,
#                                                    self.config.img_size)
#
#         # Normalize image to [-1, 1]
#         crop = 2 * ((crop / 255.) - 0.5)
#
#         return crop, proc_param, img
#
#     # def initialzation(self, img_path, json_path=None):
#     # def initialzation( self, img):
#     def initialzation(self):
#         sess = tf.Session()
#         self.model = RunModel(self.config, sess=sess)
#
#
#     # #
#     # def convert3d(self,img_org, img_dest):
#     #     # # input_img, proc_param, img = self.preprocess_image( img_path, json_path)
#     #     # input_img_org, proc_param_org, img1= self.preprocess_image( img_org)
#     #     # # Add batch dimension: 1 x D x D x 3
#     #     # input_img_org = np.expand_dims(input_img_org, 0)
#     #     #
#     #     # input_img_dest, proc_param_dest, img2 = self.preprocess_image(img_dest)
#     #     # # Add batch dimension: 1 x D x D x 3
#     #     # input_img_dest = np.expand_dims(input_img_dest, 0)
#     #     #
#     #     # joints_org, verts_org, cams_org, joints3d_org, theta_org = self.model.predict(
#     #     #     input_img_org, get_theta=True)
#     #     #
#     #     # joints_dest, verts_dest, cams_dest, joints3d_dest, theta_dest = self.model.predict(
#     #     #     input_img_dest, get_theta=True)
#     #     #
#     #     #
#     #     # return self.visualize( img1, img2, proc_param_org, proc_param_dest, joints_org[0], joints_dest[0],
#     #     #                        verts_org[0], verts_dest[0], cams_org[0], cams_dest[0])
#     #
#     #     input_img, proc_param1, img1 = self.preprocess_image(img_org)
#     #     # Add batch dimension: 1 x D x D x 3
#     #     input_img = np.expand_dims(input_img, 0)
#     #     #
#     #     # # Theta is the 85D vector holding [camera, pose, shape]
#     #     # # where camera is 3D [s, tx, ty]
#     #     # # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
#     #     # # shape is 10D shape coefficients of SMPL
#     #     #
#     #     joints1, verts1, cams1, joints3d, theta = self.model.predict(
#     #         input_img, get_theta=True)
#     #     # self.visualize( img1, proc_param, joints[0], verts[0], cams[0])
#     #
#     #     input_img, proc_param2, img2 = self.preprocess_image(img_dest)
#     #     # Add batch dimension: 1 x D x D x 3
#     #     input_img = np.expand_dims(input_img, 0)
#     #
#     #     # Theta is the 85D vector holding [camera, pose, shape]
#     #     # where camera is 3D [s, tx, ty]
#     #     # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
#     #     # shape is 10D shape coefficients of SMPL
#     #     joints2, verts2, cams2, joints3d, theta = self.model.predict(
#     #         input_img, get_theta=True)
#     #     return self.visualize(img1, img2, proc_param1, proc_param2, joints1[0], joints2[0], verts1[0], verts2[0], cams1[0],
#     #                    cams2[0])
#
#
#     def convert3d(self,img):
#         # input_img, proc_param, img = self.preprocess_image( img_path, json_path)
#         # input_img, proc_param, img = self.preprocess_image( img)
#         # # Add batch dimension: 1 x D x D x 3
#         # input_img = np.expand_dims(input_img, 0)
#         #
#         # # Theta is the 85D vector holding [camera, pose, shape]
#         # # where camera is 3D [s, tx, ty]
#         # # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
#         # # shape is 10D shape coefficients of SMPL
#         # joints, verts, cams, joints3d, theta = self.model.predict(
#         #     input_img, get_theta=True)
#         # return self.visualize( img, proc_param, joints[0], verts[0], cams[0])
#         input_img, proc_param, img = self.preprocess_image(img)
#         # Add batch dimension: 1 x D x D x 3
#         input_img = np.expand_dims(input_img, 0)
#
#         # Theta is the 85D vector holding [camera, pose, shape]
#         # where camera is 3D [s, tx, ty]
#         # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
#         # shape is 10D shape coefficients of SMPL
#         joints, verts, cams, joints3d, theta = self.model.predict(
#             input_img, get_theta=True)
#         return self.visualize(img, proc_param, joints[0], verts[0], cams[0])
#
#
#
#
#
#
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
import matplotlib.pyplot as plt
import src.config
from src.RunModel import RunModel
from PIL import Image, ImageTk
import cv2

class hmrconvert:


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

    # def init_window(self):
    #     self.master.title("GUI")
    #     self.pack(fill=BOTH, expand=1)
    #     # menu = Menu(self.master)
    #     self.master.config(menu=menu)
    #     # edit = Menu(menu)
    #     # edit.add_command(label="Choose an image", command=self.choose_img)
    #     # edit.add_command(label="Convert to 3D", command=self.conver_to_3D)
    #     # menu.add_cascade(label="Menu", menu=edit)
    #     self.img_path= None

    def visualize(self, img, proc_param, joints, verts, cam):
        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_param, verts, cam, joints, img_size=img.shape[:2])
        rend_img_overlay = self.renderer(
            vert_shifted, cam=cam_for_render, img=None, do_alpha=True, far=None,
        near=None,
        color_id=0,
        img_size=img.shape[:2])

        Image3d_org = Image.fromarray(rend_img_overlay, mode='RGBA')
        # image3d= Image.fromarray(rend_img_overlay)
        # image3d = image3d.resize((224, 224), Image.ANTIALIAS)
        # render = ImageTk.PhotoImage(image3d)
        plt.imshow(Image3d_org)
        plt.show()
        return rend_img_overlay
        # otherimg= Label(self, image=render)
        # otherimg.image=render
        # otherimg.place(x=1, y=1)

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

        # input_img, proc_param, img = self.preprocess_image( img_path, json_path)
        # input_img, proc_param, img = self.preprocess_image( img)
        # # Add batch dimension: 1 x D x D x 3
        # input_img = np.expand_dims(input_img, 0)
        #
        # # Theta is the 85D vector holding [camera, pose, shape]
        # # where camera is 3D [s, tx, ty]
        # # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # # shape is 10D shape coefficients of SMPL
        # joints, verts, cams, joints3d, theta = model.predict(
        #     input_img, get_theta=True)
        # return self.visualize( img, proc_param, joints[0], verts[0], cams[0])

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
        return self.visualize( img, proc_param, joints[0], verts[0], cams[0])

# def image_selcted(self, img_path, json_path=None):
    #     self.config = flags.FLAGS
    #     self.config(sys.argv)
    #     # Using pre-trained model, change this to use your own.
    #     self.config.load_path = src.config.PRETRAINED_MODEL
    #
    #     self.config.batch_size = 1
    #
    #     self.renderer = vis_util.SMPLRenderer(face_path=self.config.smpl_face_path)
    #     self.initialzation(img_path, json_path)

    #
    # def choose_img(self):
    #     self.img_path = filedialog.askopenfilename(filetypes=[ ('image files', ('.png', '.jpg', '.jpeg'))])
    #     if self.img_path:
    #         load = Image.open(self.img_path)
    #         load=load.resize((224, 224), Image.ANTIALIAS)
    #         render = ImageTk.PhotoImage(load)
    #
    #             # labels can be text or images
    #         img = Label(self, image=render)
    #         img.image = render
    #         img.place(x=0, y=0)

    # def conver_to_3D(self, img):
    #     # if not self.img_path or self.img_path is None:
    #     #      messagebox.showinfo("Error", "Please select an image")
    #     # else:
    #     #     self.image_selcted(self.img_path)
    #     selg.im

#     def client_exit(self):
#         exit()
# #
# root = Tk()
#
# root.geometry("400x300")
#
# app = Window(root)
#
# root.mainloop()

def main ():
    hmr= hmrconvert()
    hmr.initialzation()
    img= cv2.imread("A311.jpg")
    hmr.convert3d(img)

if __name__:
    main()