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

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

def driving_loop(wincap, preds, pause_event):
    accelerate = 1
    desired_diff = 0
    prev_angle = 0
    driving_loop_ctr = 0
    pred_ad = 0
    pred_ws = 1
    while True:
        if kb.is_pressed('/'):
            no_key()
            break

        # pause predictions when manual steering
        capture_image = np.any([kb.is_pressed(key) for key in ['up', 'down', 'left', 'right']])
        if capture_image:
            pause_event.clear()
            kb_input = ''
            for key in ['up', 'down', 'left', 'right']:
                kb_input += str(int(kb.is_pressed(key)))
            print(kb_input, time.perf_counter())
            no_key()
            # save_screen(SAVE_DIR, screen, speed, angle, kb_input)
            continue
        else:
            pause_event.set()

        # read angle
        try:
            minimap = wincap.get_screenshot(CAPTURE_MINIMAP)
        except:
            # print('compatible dlc etc')
            continue
        angle = read_angle(minimap, prev_val=prev_angle)
        if angle is None:
            # print('angle none  o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
            continue

        angle_diff = angle_diff_norm(angle, prev_angle)
        prev_angle = angle

        # speed_counter = wincap.get_screenshot(CAPTURE_SPEED)
        # try:
        #     speed = read_speed(cv2.cvtColor(speed_counter, cv2.COLOR_RGB2GRAY))
        # except ValueError:
        #     break

        if not preds.empty():
            pred = preds.get()
            if pred is None:
                no_key()
                break
            accelerate, pred_diff, pred_ws, pred_ad = pred
            # desired_diff = pred_diff * 0.7 if pred_ad != 0 else pred_diff * 0.5
            desired_diff = pred_diff * 0.5
            
            # desired_diff = angle_diff_norm(pred_angle, angle)
            # angle_diff = 0

            # print('AAAAAAAAAAAAA' if pred_ad > 0 else '------------' if pred_ad == 0 else 'DDDDDDDDDDDDDDDDDDDDD')

            # print(f'Driving loop counter: {driving_loop_ctr}')
            driving_loop_ctr = 0
            
        if np.sign(desired_diff) * np.sign(angle_diff) > 0:
            desired_diff -= angle_diff
            if np.sign(desired_diff) * np.sign(angle_diff) < 0:
                # print('overshoot:', desired_diff)
                desired_diff = 0

        # desired_diff -= angle_diff
        # print(desired_diff)
        if np.abs(desired_diff) < 4:
            desired_diff = 0
        
        steering = desired_diff

        # kb inputs
        if steering == 0:
            go_straight()
        elif steering > 0:
            turn_left()
        elif steering < 0:
            turn_right()
            
        if accelerate > 0:
            # if speed > 150:
            #     neutral()
            # else:
                speed_up()
        else:
            slow_down()
                
        driving_loop_ctr += 1
        time.sleep(0.05)
        

def predictions(model_path, wincap, preds, pause_event):
    model = get_learner(model_path)
    
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    prev_angle = 0
    angle = 0
    while True:
        pause_event.wait()
        t_start = time.perf_counter()
        if kb.is_pressed('/'):
            break

        try:
            screenshot = wincap.get_screenshot()
        except:
            continue
        
        speed_counter = cv2.cvtColor(screenshot[598:625, 1119:1187], cv2.COLOR_RGB2GRAY)
        minimap = screenshot[553:585, 112:144]

        try:
            speed = read_speed(speed_counter)
        except ValueError:
            preds.put(None)
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
        img = minimap_rotate(img, -angle/1098*360 + 90)
        ###
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
        
        pred = model.predict((img, torch.tensor(speed/100)))[0]

        accelerate = pred[0] * 100 - speed + 10
        raw_diff = int(pred[1] * 100)
        pred_ws = np.argmax(pred[3:6] ) - 1
        pred_ad = np.argmax(pred[7:]) - 1

        accelerate *= pred_ws

        # angle_diff = raw_diff
        
        pred_angle = angle + raw_diff

        preds.put((accelerate, raw_diff, pred_ws, pred_ad))
        
        t_stop = time.perf_counter()
        # print(f"Time: {(t_stop - t_start)*1000:.4f}\n\nAcc: {accelerate:.5f} \nDiff: {raw_diff:.5f} \nKb: {'{:.3f} | {:.3f} | {:.3f}'.format(*F.softmax(torch.tensor(pred[7:][::-1]), dim=0))}\n=============================")
        print(f"{(t_stop - t_start)*1000:16.2f}ms\nAcc: {accelerate:23.3f}\nStr: {raw_diff/1098*180:23.3f}\nKb WS: {'{:.3f} | {:.3f} | {:.3f}'.format(*F.softmax(torch.tensor(pred[3:6][::-1]), dim=0))}\nKb AD: {'{:.3f} | {:.3f} | {:.3f}'.format(*F.softmax(torch.tensor(pred[7:][::-1]), dim=0))}\n============================")

def test_predictions(q, w, preds, *args):
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)
    
    print('setting 5')
    preds.put((1, 5, 0, 0))
    time.sleep(3)

    print('setting -30')
    preds.put((1, -30, 0, 0))
    time.sleep(3)
    
    print('setting 0')
    preds.put((1, 0, 0, 0))
    time.sleep(3)

    print('setting 10')
    preds.put((1, 10, 0, 0))
    time.sleep(3)

    print('setting -5')
    preds.put((1, -5, 0, 0))
    time.sleep(3)

    preds.put(None)

def main():
    wincap = WindowCapture('Need for Speed™ Most Wanted')
    pause_event = Event()
    preds = Queue()

    # model_path = 'models/tiny384_100k_03off_augtfms_staticmap_v3angle_finalspeed'
    model_path = 'models/tiny384_finalspeed_27'

    t1 = Thread(target=driving_loop, args=(wincap, preds, pause_event))
    t2 = Thread(target=predictions, args=(model_path, wincap, preds, pause_event))
    # t2 = Thread(target=test_predictions, args=(model_path, wincap, preds, pause_event))

    t1.start()
    t2.start()

if __name__ == '__main__':
    main()