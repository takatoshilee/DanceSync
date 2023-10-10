import cv2
from tqdm import tqdm
import numpy as np

def write_video(fname, frames,fps, shape):
    api = cv2.CAP_FFMPEG
    code = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    output = cv2.VideoWriter(fname, api, code, fps, shape)
    with tqdm(total=len(frames), desc='Writing') as pbar:
        for frame in frames:
            output.write(frame)
            pbar.update(1)
    output.release()

ff = 1.024

path1 = 'video/caroline-choreo.mp4'
path2 = 'video/caroline-choreo-2.mp4'

cap1 = cv2.VideoCapture(path1)
cap2 = cv2.VideoCapture(path2)
fps = cap1.get(cv2.CAP_PROP_FPS)
frame_width1 = cap1.get(3)
frame_height1 = cap1.get(4)
frame_width2 = cap2.get(3)
frame_height2 = cap2.get(4)
cap1.set(cv2.CAP_PROP_POS_MSEC, ff * 1000) # second tuple now, should always be opposite

frames1 = []
frames2 = []

while 1:
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    if success1 and success2:
        # vis = np.concatenate((frame1, frame2), axis=1)
        # new_vis = cv2.resize(vis, (int(frame_width1), int(frame_height1/2)))
        # cv2.imshow("synced frames", new_vis)
        # cv2.waitKey(0)
        frames1.append(frame1)
        frames2.append(frame2)
    else:
        break

write_video('video/aligned-caroline-choreo.mp4', frames1, fps, (int(frame_width1), int(frame_height1)))
write_video('video/alignedymca-caroline-choreo-2.mp4', frames2, fps, (int(frame_width2), int(frame_height2)))