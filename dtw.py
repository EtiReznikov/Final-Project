
import cv2
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis



def main():
    org_path, dest_path = '102.mp4', 'squat103.mp4'
    org_np=videotonp(org_path)
    dest_np=videotonp(dest_path)

    path = dtw.warping_path(org_np, dest_np)
    print(path)
    dtwvis.plot_warping(org_np, dest_np, path, filename="warp.png")
def videotonp (path):
    cap=  cv2.VideoCapture(path)
    images=[]

    # cap = cv2.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret== False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(gray)
        # cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    np_frames=np.asarray(images)
    return np_frames

if __name__ == '__main__':
    main()