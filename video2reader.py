import numpy as np
import cv2
from hmr2convert import hmr2convert


org_path, dest_path= 'squat_fast.mp4', 'squat_slow.mp4'

cap_org=  cv2.VideoCapture(org_path)
i=0
while (cap_org.isOpened() and i <= 1):
    ret, frame = cap_org.read()
    height, width, layers = np.shape(frame)
    i += 1
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 1, (width, height))


paths = [org_path, dest_path]
caps = [cv2.VideoCapture(i) for i in paths]

frames = [None] * len(paths);
# gray = [None] * len(names);
ret = [None] * len(paths);
hmr = hmr2convert()
hmr.initialzation()

frame_index=0
while True:
    frames = []  # frames for one timestep
    for cap in caps:
        if cap is not None:
             ret,frame=cap.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames)==2:
        if  frame_index % 5 == 0:
            img = hmr.convert3d(frames[0], frames[1])
            img= np.array(img)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out.write(img)
    else:
        break
    frame_index+=1
    # for i,c in enumerate(cap):
    #     if c is not None:
    #         ret[i], frames[i] = c.read();
    #
    #
    # for i,f in enumerate(frames):
    #     if ret[i] is True:
    #         gray[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    #         cv2.imshow(window_titles[i], gray[i]);

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


for c in caps:
    if c is not None:
        c.release()
out.release()
cv2.destroyAllWindows()

