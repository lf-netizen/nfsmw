import threading as thr
from queue import Queue

from fastai.vision.all import *
import time
import datetime

from utils import read_angle, angle_diff_norm, preprocess_img, IMG_SIZE_X, IMG_SIZE_Y, read_speed, get_learner, save_screen, turn_left, turn_right, go_straight, speed_up, slow_down, neutral, no_key
from window_capture import WindowCapture

import cv2
import keyboard as kb
from collections import deque

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath



def driving_loop(global_run: thr.Event, scr_ready: thr.Condition, q_pred: Queue, autopilot:thr.Event):
    desired_diff = 0
    accelerate = 1
    old_angle = 0

    while True:
        global_run.wait()
        autopilot.wait()
        if global_terminate:
            no_key()
            break

        # get diff if there is any
        if not q_pred.empty():
            pred = q_pred.get()
            accelerate, desired_diff, _, pred_ad = pred

        with scr_ready:
            scr_ready.wait()
        _, speed, angle = screen_data

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


def predict(global_run: thr.Event, q_pred:Queue, autopilot:thr.Event, learn:Learner):
    while True:
        global_run.wait()
        autopilot.wait()
        if global_terminate:
            break
        
        img, speed, _ = screen_data
        img = preprocess_img(img)
        img = np.moveaxis(img, -1, 0)
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


def capture_screen(global_run: thr.Event, scr_ready: thr.Condition, wincap: WindowCapture):
    global screen_data
    while True:
        global_run.wait()
        if global_terminate:
            with scr_ready:
                time.sleep(0.2)
                scr_ready.notify_all()
            break

        try:
            img = wincap.get_screenshot()
        except:
            print('compatible dlc etc')
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        angle = read_angle(img_gray)
        speed = read_speed(img_gray)
        
        if speed is None or speed > 1000:
            terminate()
            
        screen_data = (img, speed, angle)
        with scr_ready:
            scr_ready.notify_all()


def save_screen(global_run: thr.Event, scr_ready: thr.Condition, saving: thr.Event, dir: str):
    while True:        
        global_run.wait()
        saving.wait()
        if global_terminate:
            break

        with scr_ready:
            scr_ready.wait()
        img, speed, angle = screen_data

        if angle is None:
            angle = 'None'
        else:
            angle = f'{angle:.3f}'
        if speed > 1000:
            continue

        for is_pressed in kb_input:
            kb_input += str(int(is_pressed))
        if kb_input not in ['1000', '1010', '1001', '0010', '0001', '0100', '0110', '0101']:
            continue

        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

        cv2.imwrite(dir + f'/{time}_{speed}_{angle}_{kb_input}_.png', img)


def terminate():
    global global_terminate
    global global_run
    global autopilot
    global saving
    global_terminate = True
    global_run.clear()
    autopilot.clear()
    saving.clear()
    no_key()
    print('Terminated')

screen_data = None
kb_input = [False] * 4
global_terminate = False
global_run = thr.Event()
autopilot = thr.Event()
saving = thr.Event()

THRESHOLD = 1.0
TRACKED_KEYS = ['up', 'down', 'left', 'right']
def main():
    global kb_input
    global global_run
    global autopilot
    global saving

    wincap = WindowCapture('Need for Speed™ Most Wanted')
    dir = 'images/tests/test_driving_corrections'
    q_pred = Queue()

    scr_ready = thr.Condition()
    
    # learn = get_learner_('tiny384_70k_allinp_unfrozen')
    # learn = get_learner_('tiny384_70k_allinp_v1')
    learn = get_learner('models/tiny384_70k_allinp_v3')
    # learn = get_learner('models/tiny384_70k_allinp02off_v1')

    t1 = thr.Thread(target=driving_loop, args=(global_run, scr_ready, q_pred, autopilot))
    t2 = thr.Thread(target=predict, args=(global_run, q_pred, autopilot, learn))
    t3 = thr.Thread(target=capture_screen, args=(global_run, scr_ready, wincap))
    t4 = thr.Thread(target=save_screen, args=(global_run, scr_ready, saving, dir))

    t1.start()
    # t2.start()
    t3.start()
    # t4.start()
    global_run.set()

    for i in range(2, 0, -1):
        print(i)
        
    while True:
        if kb.is_pressed('/'):
            terminate()
            break

        kb_input = [kb.is_pressed(key) for key in TRACKED_KEYS]
        if np.any(kb_input):
            no_key()
            autopilot.clear()
            print(kb_input, time.perf_counter())
        else:
            autopilot.set()

if __name__ == '__main__':
    main()


########################################################################################################################

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