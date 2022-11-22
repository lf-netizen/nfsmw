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

SAVE_DIR = 'H:/machine learning/NFSMW_v1/images/720_RGB_kb_appendix/39'

screenshot = None

def driving_loop(q_pred, wincap, make_preds):
    desired_diff = 0
    accelerate = 1
    old_angle = 0
    driving_loop_ctr = 0
    temp_dd = 0
    qlen = 4
    I = deque(maxlen=qlen)
    for _ in range(4): I.append(0)

    while True:
        if kb.is_pressed('/'):
            no_key()
            break

        # read current angle
        try:
            screen = wincap.get_screenshot()
            global screenshot
            screenshot = screen
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
            # save_screen(SAVE_DIR, screen, speed, angle, kb_input)
            continue
        else:
            make_preds.set()

        # get diff if there is any
        if not q_pred.empty():
            pred = q_pred.get()
            if desired_diff != 0: 
                temp_dd = desired_diff
                print(f'------------------------------  {desired_diff:.5f}')
            if pred is None:
                no_key()
                break
            accelerate, pred_diff, _, pred_ad = pred
            desired_diff = 0.5 * pred_diff \
                         + 0.0 * sum(I) / qlen \
                         + 0.0 * (pred_diff - I[-1])
            
            I.append(pred_diff)
            if np.sign(temp_dd) * np.sign(desired_diff) > 0:
                desired_diff += temp_dd
            temp_dd = 0
            print(f'Driving loop counter: {driving_loop_ctr}')
            driving_loop_ctr = 0
            # desired_angle = (angle + desired_diff) % 30

        # calc angle diff
        if angle is None:
            print('angle none  o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
            angle = old_angle
        angle_diff = angle_diff_norm(angle, old_angle)
        old_angle = angle
        # if np.sign(desired_diff) * np.sign(angle_diff) < 0:
        #     print('change direction +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')

        
        # check if desired diff is reached
        if np.sign(desired_diff) * np.sign(angle_diff) > 0:
            desired_diff -= angle_diff
            if np.sign(desired_diff) * np.sign(angle_diff) < 0:
                print(f'\nreached after {driving_loop_ctr} ++++++++++++++++++++++++++++++')
                print(f'overshot: {desired_diff - angle_diff}')
                # desired_diff = 0 
            # if np.sign(desired_diff) * pred_ad < 0:
            #     print('\nchange')
            #     desired_diff = 0

        # kb inputs
        if desired_diff == 0:
            go_straight()
        elif desired_diff > 0:
            turn_left()
        elif desired_diff < 0:
            turn_right()

        if accelerate > 0:
            if speed > 1700:
                neutral()
            else:
                speed_up()
        else:
            # if speed < 50:
            #     neutral()
            # else:
                slow_down()

        driving_loop_ctr += 1
        time.sleep(0.01)

THRESHOLD = 1.0
def predict(learn, q_pred, make_preds):
    for i in range(2, 0, -1):
        print(i)

    while True:
        make_preds.wait()
        t_start = time.perf_counter()
        if kb.is_pressed('/'):
            break

        img = preprocess_img(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
        speed = read_speed(cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY))

        pred = learn.predict((img, torch.tensor(speed/100)))[0]

        accelerate = pred[0] + 0.8
        raw_diff = pred[1] * 10
        pred_ws = np.argmax(pred[3:6] ) - 1
        pred_ad = np.argmax(pred[7:]) - 1
        
        # REGRESSION
        # if np.abs(raw_diff) < THRESHOLD:
        #     print('threshold ====================================')
        #     raw_diff = 0
        # angle_diff = np.ceil(raw_diff)
        # angle_diff = np.round(raw_diff)
        # angle_diff = np.sign(raw_diff)
        angle_diff = raw_diff
        # angle_diff = min(np.abs(raw_diff), 5) * np.sign(raw_diff)
        # angle_diff = np.abs(raw_diff) * pred_ad
        if pred[7] < 0 and pred[9] < 0 and pred[8] > 1:
            angle_diff = 0
            print('force straight ==============================')
        # if accelerate < 0:
        #     angle_diff = 0
        q_pred.put((accelerate, angle_diff, pred_ws, pred_ad))
        t_stop = time.perf_counter()
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
    # learn = get_learner('models/tiny384_70k_allinp_v3')
    # learn = get_learner('models/tiny384_160k_2')
    # learn = get_learner('models/tiny384_160k_02off_v1')
    learn = get_learner('models/tiny384_2heads_160k_03off_10')
    
    t1 = Thread(target=driving_loop, args=(q_pred, wincap, make_preds))
    t2 = Thread(target=predict, args=(learn, q_pred, make_preds))
      
    t1.start()
    t2.start()
    

if __name__ == '__main__':
    main()


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