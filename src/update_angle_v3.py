import numpy as np
import cv2
import os
import pickle
from threading import Thread
from queue import Queue

from utils import read_angle

img_angle = {}

def one_dir(track_path):
    img_names = next(os.walk(track_path))[2]
    if not img_names:
        return

    prev_angle = 0
    for img_name in img_names:
        try:
            img = cv2.imread(f'{track_path}/{img_name}')
            minimap = img[553:585, 112:144]
        except:
            continue
        # try:
        #     img = cv2.resize(img, (1280, 720))
        # except:
        #     print(f'resize err in {track_path}')
        #     continue

        angle = read_angle(minimap, prev_angle)

        if angle is not None:
            prev_angle = angle

        global img_angle
        img_angle[img_name] = angle
    
    print(f'Finished {track_path}')

def worker_thread(q):
    while True:
        directory = q.get()
        if directory is None:
            break
        one_dir(directory)
        q.task_done()

def run_threads(directories, num_workers):
    q = Queue()
    threads = []
    for _ in range(num_workers):
        t = Thread(target=worker_thread, args=(q,))
        t.start()
        threads.append(t)

    for directory in directories:
        q.put(directory)

    q.join()

    for i in range(num_workers):
        q.put(None)

    for t in threads:
        t.join()

def main():
    track_paths = []
    for parent_dir in ['720p_RGB', '720p_RGB_handmade_tr', '720p_RGB_kb_appendix', '720p_RGB_newdiv']:
    # for parent_dir in ['handmade_480pRGB']:
        try:
            track_paths += [f'./images/{parent_dir}/{track_path}' for track_path in next(os.walk(f'./images/{parent_dir}'))[1]]
        except StopIteration:
            pass

    run_threads(track_paths, num_workers=6)

    pickle_out = open(f'img_angles_v3.pkl', 'wb')
    pickle.dump(img_angle, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    pass
    # main()

    # pickle_in = open('img_angles_v3.pkl', 'rb')
    # angle_conv = pickle.load(pickle_in)
    # pickle_in.close()
    
    # pickle_in = open('img_angles_v3_appendix.pkl', 'rb')
    # angle_appendix = pickle.load(pickle_in)
    # pickle_in.close()

    # pickle_out = open(f'img_angles_v3.pkl', 'wb')
    # pickle.dump(angle_conv | angle_appendix, pickle_out)
    # pickle_out.close()
    