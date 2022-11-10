from threading import Thread
from queue import Queue

from fastai.vision.all import *
import time

from utils import read_angle, angle_diff_norm, preprocess_img, IMG_SIZE_X, IMG_SIZE_Y, get_learner, read_speed
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

def driving_loop(q_pred, wincap):
    pred_ws, pred_ad = 2, 0
    accelerate = 1
    while True:
        try:
            screen = wincap.get_screenshot()
        except:
            print('compatible dlc etc')
            continue
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        speed = read_speed(screen_gray)
        if speed is None:
            no_key()
            break
        
        # get diff if there is any
        if not q_pred.empty():
            pred = q_pred.get()
            if pred is None:
                no_key()
                break
            accelerate, pred_ws, pred_ad = pred
            pred_ws = np.argmax(pred_ws) - 1
            pred_ad = np.argmax(pred_ad) - 1

            
        # kb inputsne
        if pred_ad == 0:
            go_straight()
        elif pred_ad > 0:
            turn_left()
        elif pred_ad < 0:
            turn_right()

        if pred_ws > 0:
            if speed > 120:
                neutral()
            else:
                speed_up()
        else:
            # if speed < 50:
            #     neutral()
            # else:
                slow_down()
            
        if kb.is_pressed('/'):
            no_key()
            break


def predict(learn, q_pred, wincap):
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    while True:
        t_start = time.perf_counter()
        try:
            screen = wincap.get_screenshot()
        except:
            print('compatible dlc etc')
            continue
        img = preprocess_img(screen)
        img = np.moveaxis(img, -1, 0)
        speed = read_speed(cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY))

        pred = learn.predict((img, torch.tensor(speed/100)))[0]
        accelerate = pred[0] + 0.9
        pred_ws = pred[3:6]
        pred_ad = pred[7:]
        
        q_pred.put((accelerate, pred_ws, pred_ad))
        t_stop = time.perf_counter()
        print(f"Time: {(t_stop - t_start)*1000:.4f}\n\nWS: {'{:.3f} | {:.3f} | {:.3f}'.format(*pred[3:6])}\nAD: {'{:.3f} | {:.3f} | {:.3f}'.format(*pred[7:][::-1])}\n=============================")

        if kb.is_pressed('/'):
            q_pred.put(None)
            break

def test_predict(learn, q_pred, wincap):
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    print('2')
    q_pred.put((1, 2))
    time.sleep(1)

    print('-30')
    q_pred.put((1, -30))
    time.sleep(1)
    
    print('0')
    q_pred.put((1, 0))
    time.sleep(2)


    print('10')
    q_pred.put((1, 10))
    time.sleep(2)

    print('-5')
    q_pred.put((1, -5))
    time.sleep(0.5)

    q_pred.put(None)

def main():
    wincap = WindowCapture('Need for Speed™ Most Wanted')
    q_pred = Queue()
    # learn = get_learner_('tiny384_70k_allinp_unfrozen')
    # learn = get_learner_('tiny384_70k_allinp_v1')
    # learn = get_learner('models/tiny384_70k_allinp_v3')
    learn = get_learner('models/tiny384_160k_2')

    t1 = Thread(target=driving_loop, args=(q_pred, wincap))
    t2 = Thread(target=predict, args=(learn, q_pred, wincap))

    t1.start()
    t2.start()


if __name__ == '__main__':
    main()