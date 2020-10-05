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
from absl import flags
import numpy as np
from tkinter import *
import tensorflow as tf
from src.util import renderer as vis_util
from src.util import image as img_util
import src.config
from src.RunModel import RunModel
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import math
from exercise import Exercise
class hmr2convert:

    def __init__(self, master=None):
        self.config = flags.FLAGS
        self.config(sys.argv)
        # Using pre-trained model, change this to use your own.
        self.config.load_path = src.config.PRETRAINED_MODEL

        self.config.batch_size = 1

        self.renderer = vis_util.SMPLRenderer(face_path=self.config.smpl_face_path)


    def visualize_mesh(self, img_org, img_dest, proc_param_org, proc_param_dest, joints_org, joints_dest, verts_org,
                       verts_dest, cam_org, cam_dest, exercise):
        """
         Visualize two renderers mesh.
        """
        cam_for_render_org, vert_shifted_org, joints_orig_org = vis_util.get_original(
            proc_param_org, verts_org, cam_org, joints_org, img_size=img_org.shape[:2])

        cam_for_render_dest, vert_shifted_dest, joints_orig_dest = vis_util.get_original(
            proc_param_dest, verts_dest, cam_dest, joints_dest, img_size=img_dest.shape[:2])


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

        color_red = (255, 0, 0)
        color_blue = (0, 0, 255)
        thickness = 5
        radius = max(4, (np.mean(img_org.shape[:2]) * 0.01).astype(int))


        if (exercise == Exercise.squat):
            start_point = []
            start_point.append(int(joints_orig_dest[1, 0]))
            start_point.append(int(joints_orig_dest[1, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[1, 0]))
            end_point.append(int(joints_orig_org[1, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(rend_img_overlay_dest, end_point, radius, color_blue, -1)

            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                    color_red, thickness)

            start_point = []
            start_point.append(int(joints_orig_dest[4, 0]))
            start_point.append(int(joints_orig_dest[4, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[4, 0]))
            end_point.append(int(joints_orig_org[4, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(arrow_img, end_point, radius, color_blue, -1)
            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                       color_red, thickness)

            org_img_copy= rend_img_overlay_org
            output = np.concatenate((arrow_img, org_img_copy), axis=1)

        elif (exercise == Exercise.jumping_jacks):
            start_point = []
            start_point.append(int(joints_orig_dest[1, 0]))
            start_point.append(int(joints_orig_dest[1, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[1, 0]))
            end_point.append(int(joints_orig_org[1, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(rend_img_overlay_dest, end_point, radius, color_blue, -1)

            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                        color_red, thickness)

            start_point = []
            start_point.append(int(joints_orig_dest[4, 0]))
            start_point.append(int(joints_orig_dest[4, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[4, 0]))
            end_point.append(int(joints_orig_org[4, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(arrow_img, end_point, radius, color_blue, -1)
            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                        color_red, thickness)
            start_point = []
            start_point.append(int(joints_orig_dest[7, 0]))
            start_point.append(int(joints_orig_dest[7, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[7, 0]))
            end_point.append(int(joints_orig_org[7, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(arrow_img, end_point, radius, color_blue, -1)

            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                        color_red, thickness)

            start_point = []
            start_point.append(int(joints_orig_dest[10, 0]))
            start_point.append(int(joints_orig_dest[10, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[10, 0]))
            end_point.append(int(joints_orig_org[10, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(arrow_img, end_point, radius, color_blue, -1)
            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                        color_red, thickness)

            org_img_copy = rend_img_overlay_org
            output = np.concatenate((arrow_img, org_img_copy), axis=1)
        elif (exercise == Exercise.lateral_raises):
            start_point = []
            start_point.append(int(joints_orig_dest[7, 0]))
            start_point.append(int(joints_orig_dest[7, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[7, 0]))
            end_point.append(int(joints_orig_org[7, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(rend_img_overlay_dest, end_point, radius, color_blue, -1)

            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                        color_red, thickness)

            start_point = []
            start_point.append(int(joints_orig_dest[10, 0]))
            start_point.append(int(joints_orig_dest[10, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[10, 0]))
            end_point.append(int(joints_orig_org[10, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(arrow_img, end_point, radius, color_blue, -1)
            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                        color_red, thickness)

            start_point = []
            start_point.append(int(joints_orig_dest[6, 0]))
            start_point.append(int(joints_orig_dest[6, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[6, 0]))
            end_point.append(int(joints_orig_org[6, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(arrow_img, end_point, radius, color_blue, -1)
            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                        color_red, thickness)

            start_point = []
            start_point.append(int(joints_orig_dest[11, 0]))
            start_point.append(int(joints_orig_dest[11, 1]))
            start_point = tuple(start_point)

            end_point = []
            end_point.append(int(joints_orig_org[11, 0]))
            end_point.append(int(joints_orig_org[11, 1]))
            end_point = tuple(end_point)

            arrow_img = cv2.circle(arrow_img, end_point, radius, color_blue, -1)
            arrow_img = cv2.arrowedLine(arrow_img, start_point, end_point,
                                        color_red, thickness)

            org_img_copy = rend_img_overlay_org

            output = np.concatenate((arrow_img, org_img_copy), axis=1)

        Image3d_org=crop_img(rend_img_overlay_org, joints_org)
        Image3d_dest=crop_img(rend_img_overlay_dest, joints_dest)

        Image3d_dest= move_img( Image3d_dest, joints_org, joints_dest)
        Image3d_org = Image.fromarray(Image3d_org , mode='RGBA')
        Image3d_dest = Image.fromarray(Image3d_dest, mode='RGBA')


        mask = Image.new("L", Image3d_dest.size, 128)

        im = Image.composite(Image3d_dest, Image3d_org, mask)
        im_np= np.asarray(im)
        print(im_np.shape)
        print(output.shape)
        output = np.concatenate((im_np, output), axis=1)
        return output


    def visualize_arroweds(self, img_org, img_dest, proc_param_org, proc_param_dest, joints_org, joints_dest, verts_org,
                           verts_dest, cam_org, cam_dest, exercise):

        cam_for_render_org, vert_shifted_org, joints_orig_org = vis_util.get_original(
            proc_param_org, verts_org, cam_org, joints_org, img_size=img_org.shape[:2])

        cam_for_render_dest, vert_shifted_dest, joints_orig_dest = vis_util.get_original(
            proc_param_dest, verts_dest, cam_dest, joints_dest, img_size=img_dest.shape[:2])


        color_red = (255, 0, 0)
        color_blue = (0, 0, 255)
        # Line thickness of 9 px
        thickness = 5
        radius = max(4, (np.mean(img_org.shape[:2]) * 0.01).astype(int))


        start_point =[]
        start_point.append(int(joints_orig_org[1,0]))
        start_point.append(int(joints_orig_org[1,1]))
        start_point= tuple(start_point)

        end_point = []
        end_point.append(int(joints_orig_dest[1, 0]))
        end_point.append(int(joints_orig_dest[1, 1]))
        end_point = tuple(end_point)


        skel_img = cv2.circle(img_org, end_point, radius, color_blue, -1)

        start_point = []
        start_point.append(int(joints_orig_org[4, 0]))
        start_point.append(int(joints_orig_org[4, 1]))
        start_point = tuple(start_point)

        end_point = []
        end_point.append(int(joints_orig_dest[4, 0]))
        end_point.append(int(joints_orig_dest[4, 1]))
        end_point = tuple(end_point)


        skel_img = cv2.circle(skel_img, end_point, radius, color_blue, -1)
        skel_img = cv2.arrowedLine(skel_img, start_point, end_point,
                                   color_red, thickness)
        skel_img = cv2.circle(skel_img, start_point, radius, color_blue, -1)

        output = np.concatenate((skel_img, img_dest), axis=1)

        return output

    def preprocess_image(self, img):

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

        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                   self.config.img_size)

        # Normalize image to [-1, 1]dest [[ 81.447205 202.8527  ]]

        crop = 2 * ((crop / 255.) - 0.5)

        return crop, proc_param, img

    def initialzation(self):
        sess = tf.Session()
        self.model = RunModel(self.config, sess=sess)

    def convert3d(self,img_org, img_dest, flag, exercise):

        #Source image parameters
        input_img_org, proc_param_org, img1 = self.preprocess_image(img_org)
        # Add batch dimension: 1 x D x D x 3
        input_img_org = np.expand_dims(input_img_org, 0)
        joints_org, verts_org, cams_org, joints3d_org, theta_org = self.model.predict(
            input_img_org, get_theta=True)

        # target image parameters
        input_img_dest, proc_param_dest, img2 = self.preprocess_image(img_dest)
        # Add batch dimension: 1 x D x D x 3
        input_img_dest = np.expand_dims(input_img_dest, 0)
        joints_dest, verts_dest, cams_dest, joints3d_dest, theta_dest = self.model.predict(
            input_img_dest, get_theta=True)


        if flag: #return mesh - 3D visual feedback
            return self.visualize_mesh(img1, img2, proc_param_org, proc_param_dest, joints_org[0], joints_dest[0],
                                       verts_org[0], verts_dest[0], cams_org[0], cams_dest[0], exercise)
        else:  #return 2D visual feedback
            return self.visualize_arroweds(img1, img2, proc_param_org, proc_param_dest, joints_org[0], joints_dest[0],
                                           verts_org[0], verts_dest[0], cams_org[0], cams_dest[0], exercise)



    def get_joints3d(self,img):
        "return the 3D joints for given image"
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


def get_hegiht(joints):
    """
    return the higeht of person accroding to the joints from ankels to head
    """
    joints=np.reshape(joints, (1,19,2))
    right_ankle= joints[:,0]
    left_ankle= joints[:,5]
    right_ankle_y=right_ankle[0][1]
    left_ankle_y=left_ankle[0][1]
    if (right_ankle_y< left_ankle_y):
        ankle=right_ankle[0]
    else:
        ankle=left_ankle[0]

    head= joints[:,13][0]

    height= math.sqrt((head[1]-ankle[1])**2+(head[0]-ankle[0])**2)
    return height, right_ankle[0], head


def crop_img(img, joints):
    """
    Cut the image so that the image will include only the figure without unnecessary background
    """
    height, ankle, head = get_hegiht(joints)
    shape = np.shape(img)

    ankle*= (shape[0] / 224)
    head*= (shape[0] / 224)

    crop_h_1 = shape[0]
    crop = img[:crop_h_1, :, :]
    scale = shape[1] / shape[0]
    crop_width = scale * (shape[0] - np.shape(crop)[0])
    crop_width = int(crop_width / 2)
    crop_width_lim = shape[1] - crop_width
    crop = crop[:, crop_width:crop_width_lim, :]
    img_crop = cv2.resize(crop, (shape[1], shape[0]),
                          interpolation=cv2.INTER_AREA)
    return img_crop


def move_img(img_dest, joints_org, joints_dest):
    """
    Translate the target image on the source image according to the right ankle so that the two figures will "stand" in the same place in the image.
    """

    im2_dest_shape= np.shape(img_dest)

    org_height, org_ankle, org_head= get_hegiht(joints_org)
    dest_height, dest_ankle, dest_head = get_hegiht(joints_dest)

    t_x=int(org_ankle[1]-dest_ankle[1])
    t_y=int(org_ankle[0]-dest_ankle[0])

    #transform to bottom and right
    if t_x<0 and t_y<0:
        img_dest = cv2.copyMakeBorder( img_dest, 0,-1*t_x ,0 , -1*t_y, cv2.BORDER_CONSTANT)
        img_dest = img_dest[-1*t_x:, -1*t_y:, :]
    #transform to top and right
    elif t_x >= 0 and t_y < 0:
        img_dest = cv2.copyMakeBorder( img_dest, t_x,0 ,0, -1*t_y, cv2.BORDER_CONSTANT)
        img_dest = img_dest[0:(im2_dest_shape[0]-t_x), -1*t_y: , :]
    # transform to bottom and left
    elif t_x < 0 and t_y >= 0:
        img_dest = cv2.copyMakeBorder( img_dest, 0, -1*t_x, t_y, 0, cv2.BORDER_CONSTANT)
        img_dest = img_dest[-1*t_x: , 0:(im2_dest_shape[1]-t_y), :]
    # transform to top and left
    elif t_x >= 0 and t_y >= 0:
        img_dest = cv2.copyMakeBorder( img_dest, t_x, 0, t_y, 0, cv2.BORDER_CONSTANT)
        img_dest = img_dest[ 0:im2_dest_shape[0]-t_x,  0:im2_dest_shape[1]-t_y, :]
    return img_dest





def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=3, length_includes_head=True)

