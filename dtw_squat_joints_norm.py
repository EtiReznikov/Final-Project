# from dtw import dtw
import numpy as np
import cv2
from hmr_rendrer import hmr_rendrer
from dtaidistance import dtw

from hmr2convert import hmr2convert
import math

org_path, dest_path= 'squat103.mp4', 'squat102.mp4'
hmr = hmr2convert()
hmr.initialzation()

def get_joints(path, hmr):
    joints_video=[]
    cap = cv2.VideoCapture(path)

    # hmr = hmr_rendrer()
    # hmr.initialzation()
    i=0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % 5 == 0:
            # frames.append(frame)
            joints= hmr.get_joints3d(frame)
            joints_video.append(joints)

        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return joints_video

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




def get_angles(joints3d):
    z_vec= np.array([0,0,1])
    rows, cols , dims= np.shape(joints3d)
    angles= np.zeros((rows, cols, 1))
    for i in range(rows):
        for j in range(cols):
            angles[i,j]= angle_between(z_vec, joints3d[i,j])
    return  angles


def get_norm_angles(joints):
    norm_joints=[]

    for i in range (0,np.shape(joints)[0]):
        v=[]
        v.append(joints[i,1,:])
        v.append(joints[i,4,:])
        norm=unit_vector(v)
        norm_joints.append(norm)
    return (norm_joints)

def dtw_for_angles(seq1, seq2):
    dtw_joints=[]
    len1= len(seq1)
    len2= len(seq2)
    seq1_v=[]
    for j in range (0,6):
        joint1= seq1[:,j,:]
        seq1_v.append(joint1)
    joint1= np.reshape(joint1, len1)
    # print(np.shape(joint1))
    seq2_v=[]
    for j in range (0,6):
        joint2= seq2[:,j,:]
        seq2_v.append(joint2)
    joint2= np.reshape(joint2, len2)

    joint1= unit_vector(joint1)
    joint2= unit_vector(joint2)
    # print(np.shape(joint2))
    # d=dtw(joint1, joint2)
    # manhattan_distance = lambda joint1, joint2 : np.abs(joint1 - joint2)
    # d, cost_matrix, acc_cost_matrix, path = dtw(joint1, joint2, dist=manhattan_distance)
    #
    d=  dtw.warping_path(joint1,  joint2 )
    d=np.array(d)
    # dtw_joints.append(d)
    # dtw_joints=np.array(dtw_joints)
    # return dtw_joints
    return d

# def dtw_for_angles(seq1, seq2):
#     dtw_joints=[]
#     len1= len(seq1)
#     len2= len(seq2)
#     seq1_v=[]
#     for j in range (0,19):
#         joint1= seq1[:,j,:]
#         seq1_v.append(joint1)
#     joint1= np.reshape(joint1, len1)
#     # print(np.shape(joint1))
#     seq2_v=[]
#     for j in range (0,19):
#         joint2= seq2[:,j,:]
#         seq2_v.append(joint2)
#     joint2= np.reshape(joint2, len2)
#
#     joint1= unit_vector(joint1)
#     joint2= unit_vector(joint2)
#     # print(np.shape(joint2))
#     # d=dtw(joint1, joint2)
#     # manhattan_distance = lambda joint1, joint2 : np.abs(joint1 - joint2)
#     # d, cost_matrix, acc_cost_matrix, path = dtw(joint1, joint2, dist=manhattan_distance)
#     #
#     d=  dtw.warping_path(joint1,  joint2 )
#     d=np.array(d)
#     # dtw_joints.append(d)
#     # dtw_joints=np.array(dtw_joints)
#     # return dtw_joints
#     return d








def change_video(dtw_array, video_org, video_dest):
    # dtw_dest=dtw_array[0,0]
    # dtw_org=dtw_array[0,1]
    dtw_dest=dtw_array[0,1]
    dtw_org=dtw_array[0,0]
    dtw_ind=0
    print(np.shape(dtw_array)[0])
    cap_org = cv2.VideoCapture(video_org)
    i = 0
    while (cap_org.isOpened() and i <= 1):
        ret, frame = cap_org.read()
        height, width, layers = np.shape(frame)
        i += 1
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 1, (width, height))

    paths = [video_dest, video_org]
    caps = [cv2.VideoCapture(i) for i in paths]

    frames = [None] * len(paths);
    # gray = [None] * len(names);
    ret = [None] * len(paths);
    # hmr = hmr2convert()
    # hmr.initialzation()

    frame_index = 0
    # while True :
    while dtw_ind < np.shape(dtw_array)[0]:
        # current_dtw_dest = dtw_array[dtw_ind,0]
        # current_dtw_org = dtw_array[dtw_ind,1]
        current_dtw_dest = dtw_array[dtw_ind,1]
        current_dtw_org = dtw_array[dtw_ind,0]
        if (dtw_ind==0):
            frames = []
            ret, frame = caps[0].read()
            if not ret:
                break
            frames.append(frame)

            ret, frame = caps[1].read()
            if not ret:
                break
            frames.append(frame)
        else:
            old_frames= frames
            frames = []  # frames for one timestep
            # current_dtw_r = dtw_array[0, dtw_ind]
            # dtw_ind+=1

            # dtw_r= current_dtw_r

            # current_dtw_l = dtw_array[0, dtw_ind]
            # dtw_ind += 1
            if (current_dtw_dest == dtw_dest):
                print("true left")
                frames.append((old_frames[0]))
            else:
                ret, frame = caps[0].read()
                if not ret:
                    print("dtw_l")
                    break
                frames.append(frame)

            if (current_dtw_org == dtw_org):
                print("true right")
                frames.append((old_frames[1]))
            else:
                ret, frame = caps[1].read()
                if not ret:
                    print("dtw_r")
                    break
                frames.append(frame)
            # dtw_l = current_dtw_l
        dtw_org = current_dtw_org
        dtw_dest = current_dtw_dest
        print(dtw_org, " , ", dtw_dest)
        dtw_ind += 1

        if len(frames) == 2:
            if frame_index % 5 == 0:
                img = hmr.convert3d(frames[1], frames[0])
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                out.write(img)
        else:
            print("frame==1")
            break
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("while", dtw_ind)
    for c in caps:
        if c is not None:
            c.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    joints1 = get_joints(org_path, hmr)
    joints1 = np.array(joints1)

    angles1 = get_angles(joints1)

    print(np.shape(angles1))
    # norm_angles_1=get_norm_angles(angles1)

    joints2 = get_joints(dest_path, hmr)
    joints2 = np.array(joints2)

    angles2 = get_angles(joints2)
    # norm_angles_2=get_norm_angles(angles2)

    print("-------------------------")
    dtw_array = dtw_for_angles(angles1, angles2)
    # dtw_array = dtw_for_angles(norm_angles_1, norm_angles_2)
    change_video(dtw_array, org_path, dest_path)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angles_unit_vector(vector):
    """ Returns the unit vector of the vector.
     more wieght to knees"""
    sum=0
    for ang in vector:
        sum+= ang**2
    sum+=vector[4]**2
    sum+=vector[1]**2
    return math.sqrt(sum)
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

if __name__ == '__main__':
    main()

