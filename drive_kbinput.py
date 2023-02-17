from threading import Thread, Event
from queue import Queue

from fastai.vision.all import *
import torch.nn.functional as F
import time

from utils import read_angle, angle_diff_norm, preprocess_img, IMG_SIZE_X, IMG_SIZE_Y, read_speed, get_learner, save_screen, minimap_rotate, CAPTURE_MINIMAP, CAPTURE_SPEED, turn_left, turn_right, go_straight, speed_up, slow_down, neutral, no_key
from window_capture import WindowCapture

import cv2
import keyboard as kb
from collections import deque

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
        

def predictions(model_path, wincap):
    model = get_learner(model_path)
    
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    prev_angle = 0
    angle = 0
    while True:
        t_start = time.perf_counter()
        if kb.is_pressed('/'):
            no_key()
            break
        # capture_image = np.any([kb.is_pressed(key) for key in ['up', 'down', 'left', 'right']])
        # if capture_image:
        #     kb_input = ''
        #     for key in ['up', 'down', 'left', 'right']:
        #         kb_input += str(int(kb.is_pressed(key)))
        #     print(kb_input, time.perf_counter())
        #     no_key()
        #     continue
        
        try:
            screenshot = wincap.get_screenshot()
        except:
            continue
        
        speed_counter = cv2.cvtColor(screenshot[598:625, 1119:1187], cv2.COLOR_RGB2GRAY)
        minimap = screenshot[553:585, 112:144]

        try:
            speed = read_speed(speed_counter)
        except ValueError:
            break

        angle_temp = read_angle(minimap, prev_val=prev_angle)

        if angle_temp is None:
            continue

        prev_angle = angle
        angle = angle_temp

        img = preprocess_img(screenshot)
        # for static minimap
        if angle_temp is not None:
            angle = angle_temp
        img = minimap_rotate(img, angle)
        ###
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
        
        pred = model.predict((img, torch.tensor(speed/100)))[0]

        pred_ws = np.argmax(pred[3:6] ) - 1
        pred_ad = np.argmax(pred[7:]) - 1
        

        steering = pred_ad
        accelerate = pred_ws
        if steering == 0:
            go_straight()
        elif steering > 0:
            turn_left()
        elif steering < 0:
            turn_right()
        if accelerate > 0:
            if speed > 150:
                neutral()
            else:
                speed_up()
        else:
            slow_down()


        t_stop = time.perf_counter()
        print(f"Time: {(t_stop - t_start)*1000:.4f}\n \nAD: {'{:.3f} | {:.3f} | {:.3f}'.format(*F.softmax(torch.tensor(pred[7:][::-1]), dim=0))}\n{'AAAAAAAAAAAAA' if pred_ad > 0 else '------------' if pred_ad == 0 else 'DDDDDDDDDDDDDDDDDDDDD'}\nWS: {'{:.3f} | {:.3f} | {:.3f}'.format(*F.softmax(torch.tensor(pred[3:6][::-1]), dim=0))}\n=============================")

def main():
    wincap = WindowCapture('Need for Speed™ Most Wanted')
    model_path = 'models/tiny384_100k_03off_augtfms_staticmap_v3angle'
    
    predictions(model_path, wincap)

if __name__ == '__main__':
    main()