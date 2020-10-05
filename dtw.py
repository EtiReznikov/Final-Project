# from dtw import dtw
import sys
import numpy as np
import cv2
from dtaidistance import dtw
from hmr2convert import hmr2convert

import timeit

org_path_squat='videos/squat_good.mp4'
dest_path_squat='videos/squat_bad.mp4'

org_path_jumping_jacks='videos/jumping_jacks_good.mp4'
dest_path_jumping_jacks='videos/jumping_jacks_bad.mp4'

org_path_lateral_raises='videos/lateral_raises_good.mp4'
dest_path_lateral_raises='videos/lateral_raises_bad.mp4'


hmr = hmr2convert()
hmr.initialzation()

def get_joints(path):
    """
    get all the joints to each frame
    """
    joints_video=[]
    cap = cv2.VideoCapture(path)

    i=0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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



def get_angel_between_vectors(ankle, knee, hip):
    """
    return the angle between two vectors
    """
    vec_ankle_knee= ankle-knee
    vec_hip_knee = hip - knee
    return angle_between(vec_ankle_knee, vec_hip_knee)


def get_squat_angles(joints3d):
    """
    return the average of the knees angles to each frame
    """
    rows, cols , dims= np.shape(joints3d)
    angles= np.zeros((rows))
    for i in range(rows):
        angles[i] += get_angel_between_vectors(joints3d[i, 0], joints3d[i, 1], joints3d[i, 2])
        angles[i] += get_angel_between_vectors(joints3d[i, 5], joints3d[i, 4], joints3d[i, 3])
        angles[i] /=2
    return  angles

def get_jumping_jacks_angles(joints3d):
    """
    return the average of the armpits and the groin angles to each frame
    """
    rows, cols , dims= np.shape(joints3d)
    angles= np.zeros((rows))
    for i in range(rows):

        angles[i] += get_angel_between_vectors(joints3d[i, 2], joints3d[i, 8], joints3d[i, 7])
        angles[i] += get_angel_between_vectors(joints3d[i, 3], joints3d[i, 9], joints3d[i, 10])
        angles[i] += get_angel_between_vectors(joints3d[i, 1], joints3d[i, 12], joints3d[i, 4])
        angles[i] /= 3

    return  angles

def get_lateral_raises_angles(joints3d):
    """
    return the average of the armpits angles to each frame
    """
    rows, cols , dims= np.shape(joints3d)
    angles= np.zeros((rows))
    for i in range(rows):
        angles[i] += get_angel_between_vectors(joints3d[i, 2], joints3d[i, 8], joints3d[i, 7])
        angles[i] += get_angel_between_vectors(joints3d[i, 3], joints3d[i, 9], joints3d[i, 10])
        angles[i] /= 2

    return  angles



def dtw_path(seq1, seq2):
    """
    return the dtw path
    """
    path=  dtw.warping_path(seq1,  seq2)
    path=np.array(path)
    return path


def dtw_on_videos(dtw_array, org_path, dest_path, is_mesh, exercise):
    """
    change the original videos such that the pace of both exercise it the target and the source will be similar.
    We did that by duplicate the frames which the pose is preform faster accroding to dtw path.
    The output is video with the required visual feedback
    """
    cap_org = cv2.VideoCapture(org_path)
    i = 0
    while (cap_org.isOpened() and i <= 1):
        ret, frame = cap_org.read()
        height, width, layers = np.shape(frame)
        i += 1
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # if is_mesh:
    #     out = cv2.VideoWriter('output.avi', fourcc, 1, (width, height))
    # else:
    #     out = cv2.VideoWriter('output.avi', fourcc, 1, (2*width, height))

    out = cv2.VideoWriter('output.avi', fourcc, 1, (3*width, height))
    paths = [org_path, dest_path]
    caps = [cv2.VideoCapture(i) for i in paths]

    dtw_ind=0
    dtw_org=dtw_array[dtw_ind][0]
    dtw_dest=dtw_array[dtw_ind][1]

    frame_index_org = 0
    frame_index_dest = 0
    while dtw_ind < np.shape(dtw_array)[0]:
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

            frame_index_org += 1
            frame_index_dest += 1
        else:
            old_frames = frames
            frames = []
            dtw_org_old = dtw_org
            dtw_org = dtw_array[dtw_ind][0]

            if dtw_org_old == dtw_org:
                frames.append(old_frames[0])
            else:
                ret, frame = caps[0].read()
                if not ret:
                    break
                frames.append(frame)
            frame_index_org+=1

            dtw_dest_old = dtw_dest
            dtw_dest = dtw_array[dtw_ind][1]

            if dtw_dest_old == dtw_dest:
                frames.append(old_frames[1])
            else:
                ret, frame = caps[1].read()
                if not ret:
                    break
                frames.append(frame)

            frame_index_dest += 1

        dtw_ind+=1

        if len(frames)==2:
            frame_org = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
            frame_dest = cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB)
            img = hmr.convert3d(frame_org, frame_dest, is_mesh, exercise)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            out.write(img)
        else:
            break;
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for c in caps:
        if c is not None:
            c.release()
    out.release()
    cv2.destroyAllWindows()


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


from exercise import Exercise
def main():
    exercise = Exercise.squat
    start = timeit.default_timer()

    if (exercise == Exercise.squat):
        org_path=org_path_squat
        dest_path=dest_path_squat
        joints_org = get_joints(org_path)
        joints_org = np.array(joints_org)

        joints_dest = get_joints(dest_path)
        joints_dest = np.array(joints_dest)

        angles_org= get_squat_angles(joints_org)
        angles_dest = get_squat_angles(joints_dest)
    elif (exercise == Exercise.jumping_jacks):
        org_path=org_path_jumping_jacks
        dest_path=dest_path_jumping_jacks
        joints_org = get_joints(org_path)
        joints_org = np.array(joints_org)

        joints_dest = get_joints(dest_path)
        joints_dest = np.array(joints_dest)

        angles_org=get_jumping_jacks_angles(joints_org)
        angles_dest=get_jumping_jacks_angles(joints_dest)
    elif (exercise == Exercise.lateral_raises):
        org_path=org_path_lateral_raises
        dest_path=dest_path_lateral_raises
        joints_org = get_joints(org_path)
        joints_org = np.array(joints_org)

        joints_dest = get_joints(dest_path)
        joints_dest = np.array(joints_dest)

        angles_org=get_lateral_raises_angles(joints_org)
        angles_dest= get_lateral_raises_angles(joints_dest)
    dtw_array= dtw_path(angles_org, angles_dest)

    is_mesh= True
    dtw_on_videos (dtw_array, org_path, dest_path, is_mesh,exercise)
    stop = timeit.default_timer()

    print('Time: ', stop - start)

if __name__ == '__main__':
    main()

