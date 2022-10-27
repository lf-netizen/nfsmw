from threading import Thread, Event
from queue import Queue

from fastai.vision.all import *
import time

from utils import read_angle, angle_diff_norm, preprocess_img, IMG_SIZE_X, IMG_SIZE_Y, read_speed, get_learner, save_screen
from window_capture import WindowCapture

import cv2
import keyboard as kb
from collections import deque

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def turn_left():
    kb.press('a')
    kb.release('d')
def turn_right():
    kb.release('a')
    kb.press('d')
def go_straight():
    kb.release('a')
    kb.release('d')

def speed_up():
    kb.press('w')
    kb.release('s')
def slow_down():
    kb.release('w')
    kb.press('s')
def neutral():
    kb.release('w')
    kb.release('s')

def no_key():
    kb.release('w')
    kb.release('s')
    kb.release('a')
    kb.release('d')

def driving_loop(q_pred, wincap, make_preds):
    desired_diff = 0
    accelerate = 1
    old_angle = 0

    make_preds.set()
    while True:
        if kb.is_pressed('/'):
            no_key()
            break

        # read current angle
        try:
            screen = wincap.get_screenshot()
        except:
            print('compatible dlc etc')
            continue
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        angle = read_angle(screen_gray)
        speed = read_speed(screen_gray)
        if speed is None:
            no_key()
            break
        
        capture_image = np.any([kb.is_pressed(key) for key in ['up', 'down', 'left', 'right']])
        if capture_image:
            make_preds.clear()
            kb_input = ''
            for key in ['up', 'down', 'left', 'right']:
                kb_input += str(int(kb.is_pressed(key)))
            print(kb_input, time.perf_counter())
            no_key()
            continue
        else:
            make_preds.set()

        # get diff if there is any
        if not q_pred.empty():
            pred = q_pred.get()
            if pred is None:
                no_key()
                break
            accelerate, desired_diff, _, pred_ad = pred
            # desired_angle = (angle + desired_diff) % 30

        # calc angle diff
        if angle is None:
            angle = old_angle
        angle_diff = angle_diff_norm(angle, old_angle)
        old_angle = angle
        
        # check if desired diff is reached
        if np.sign(desired_diff)*np.sign(angle_diff) > 0:
            desired_diff -= angle_diff
            if np.sign(desired_diff)*np.sign(angle_diff) < 0:
                desired_diff = 0
            if np.sign(desired_diff)*pred_ad < 0:
                print('\nchange')
                desired_diff = 0

        # kb inputs
        if desired_diff == 0:
            go_straight()
        elif desired_diff > 0:
            turn_left()
        elif desired_diff < 0:
            turn_right()

        if accelerate > 0:
            if speed > 120:
                neutral()
            else:
                speed_up()
        else:
            # if speed < 50:
            #     neutral()
            # else:
                slow_down()


        
def driving_loop_simple(q_pred, wincap):
    desired_diff = 0
    accelerate = 1
    while True:
        # read current angle
        try:
            screen = wincap.get_screenshot()
        except:
            print('compatible dlc etc')
            continue
        
        
        # get diff if there is any
        if not q_pred.empty():
            pred = q_pred.get()
            if pred is None:
                no_key()
                break
            accelerate, desired_diff, _ = pred
            # desired_angle = (angle + desired_diff) % 30

        # kb inputs
        if desired_diff == 0:
            go_straight()
        elif desired_diff > 0:
            turn_left()
        elif desired_diff < 0:
            turn_right()

        if accelerate > 0:
            speed_up()
        else:
            slow_down()

        if kb.is_pressed('/'):
            no_key()
            break

THRESHOLD = 1
def predict(learn, q_pred, wincap, make_preds):
    for i in range(2, 0, -1):
        print(i)

    while True:
        make_preds.wait()
        if kb.is_pressed('/'):
            break
        # if capture_image:
        #     continue

        t_start = time.perf_counter()
        try:
            screen = wincap.get_screenshot()
        except:
            print('compatible dlc etc')
            continue
        img = preprocess_img(screen)
        img = np.moveaxis(img, -1, 0)
        speed = read_speed(cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY))

        t_start = time.perf_counter()
        pred = learn.predict((img, torch.tensor(speed/100)))[0]
        t_stop = time.perf_counter()

        accelerate = pred[0] + 0.8
        raw_diff = pred[1] * 10
        pred_ws = np.argmax(pred[3:6])-1
        pred_ad = np.argmax(pred[7:])-1
        
        # REGRESSION
        if np.abs(raw_diff) < THRESHOLD:
            raw_diff = 0
        # angle_diff = np.ceil(raw_diff)
        # angle_diff = np.round(raw_diff)
        # angle_diff = np.sign(raw_diff)
        angle_diff = raw_diff
        # angle_diff = min(raw_diff**2, 8) * np.sign(raw_diff)
        
        q_pred.put((accelerate, angle_diff, pred_ws, pred_ad))
        print(f"Time: {(t_stop - t_start)*1000:.4f}\n\nAcc: {accelerate:.5f} \nDiff: {raw_diff:.5f} \nKb: {'{:.3f} | {:.3f} | {:.3f}'.format(*pred[7:][::-1])}\n=============================")


def test_predict(learn, q_pred, wincap):
    for i in range(2, 0, -1):
        print(i)

    print('2')
    q_pred.put((1, 2, 0))

    print('-30')
    q_pred.put((1, -30, 0))
    
    print('0')
    q_pred.put((1, 0, 0))


    # print('10')
    # q_pred.put((1, 10, 0))

    print('-5')
    q_pred.put((1, -5, 0))

    q_pred.put(None)

def main():
    wincap = WindowCapture('Need for Speed™ Most Wanted')
    q_pred = Queue()
    make_preds = Event()
    # learn = get_learner_('tiny384_70k_allinp_unfrozen')
    # learn = get_learner_('tiny384_70k_allinp_v1')
    learn = get_learner('models/tiny384_70k_allinp_v3')
    # learn = get_learner('models/tiny384_70k_allinp02off_v1')

    t1 = Thread(target=driving_loop, args=(q_pred, wincap, make_preds))
    t2 = Thread(target=predict, args=(learn, q_pred, wincap, make_preds))

    t1.start()
    t2.start()


if __name__ == '__main__':
    main()
