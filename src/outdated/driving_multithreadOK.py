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

SAVE_DIR = 'H:/machine learning/NFSMW_v1/images/720_RGB_kb_appendix/39'

screenshot = None

def driving_loop(q_pred, wincap, make_preds):
    desired_diff = 0
    accelerate = 1
    old_angle = 0
    driving_loop_ctr = 0
    temp_dd = 0
    qlen = 4
    pred_ad = 0
    I = deque(maxlen=qlen)
    for _ in range(4): I.append(0)
    steering = 0
    break_flag = True

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

        speed_counter = cv2.cvtColor(screen[598:625, 1119:1187], cv2.COLOR_RGB2GRAY)
        try:
            speed = read_speed(speed_counter)
        except:

            no_key()
            break
        
        minimap = screenshot[553:585, 112:144]
        angle = read_angle(minimap, prev_val=old_angle)

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
            desired_diff = pred_diff
            # accelerate = pred_speed - speed
            
            # if np.sign(
            # temp_dd) * np.sign(desired_diff) > 0:
            #     desired_diff += temp_dd
            # temp_dd = 0
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

        
        # desired_diff -= angle_diff
        # steering = 1.0 * desired_diff \
                # + 0.0 * sum(I) / qlen \
                # + 0.0 * (angle_diff - I[-1])
        # I.append(angle_diff)

        # check if desired diff is reached
        if np.sign(desired_diff) * np.sign(angle_diff) > 0:
            desired_diff -= angle_diff
            if np.sign(desired_diff) * np.sign(angle_diff) < 0:
                # print(f'\nreached after {driving_loop_ctr} ++++++++++++++++++++++++++++++')
                print(f'overshot: {desired_diff - angle_diff}')
                desired_diff = 0 
            # if np.sign(desired_diff) * pred_ad < 0:
            #     print('\nchange')
            #     desired_diff = 0
        steering = desired_diff
        
        # kb inputs
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
            # if speed < 50:
            # if steering != 0:
                # neutral()
            # else:
                # if break_flag % 4:
                #     neutral()
                #     break_flag = 0
                # else:
                    slow_down()
                # break_flag += 1

        print(desired_diff)
        driving_loop_ctr += 1
        time.sleep(0.005)

THRESHOLD = 1.0
def predict(learn, q_pred, make_preds):
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    angle = 0
    while True:
        make_preds.wait()
        t_start = time.perf_counter()
        if kb.is_pressed('/'):
            break

        img = preprocess_img(screenshot)
        speed_counter = cv2.cvtColor(screenshot[598:625, 1119:1187], cv2.COLOR_RGB2GRAY)
        speed = read_speed(speed_counter)
        
        # for static minimap
        minimap = screenshot[553:585, 112:144]
        angle_temp = read_angle(minimap, prev_val=angle)
        if angle_temp is not None:
            angle = angle_temp
        img = minimap_rotate(img, angle)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
        
        pred = learn.predict((img, torch.tensor(speed/100)))[0]

        accelerate = pred[0] * 100 - speed
        raw_diff = np.round(pred[1] * 100)
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
        # if pred[7] < 0 and pred[9] < 0 and pred[8] > 1:
        #     angle_diff = 0
        #     print('force straight ==============================')
        # if accelerate < 0:
        #     angle_diff = 0
        q_pred.put((accelerate, angle_diff, pred_ws, pred_ad))
        t_stop = time.perf_counter()
        print(f"Time: {(t_stop - t_start)*1000:.4f}\n\nAcc: {accelerate:.5f} \nDiff: {raw_diff:.5f} \nKb: {'{:.3f} | {:.3f} | {:.3f}'.format(*F.softmax(torch.tensor(pred[7:][::-1]), dim=0))}\n=============================")


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
    # learn = get_learner('models/tiny384_2heads_160k_03off_10')
    # learn = get_learner('models/tiny384_2heads_aug_6')
    # learn = get_learner('models/tiny384_2heads_newdiv_4')
    # learn = get_learner('models/small_2head_8')
    # learn = get_learner('models/tiny384_2heads_newdiv_aug_4')
    # learn = get_learner('models/tiny384_augtfms_staticmap_20')
    # learn = get_learner('models/tiny384_2heads_aug_staticmap_16')
    # learn = get_learner('models/tiny384_100k_03off_augtfms_staticmap_v3angle')
    learn = get_learner('models/tiny384_100k_03off_augtfms_staticmap_v3angle_finalspeed')
    
    t1 = Thread(target=driving_loop, args=(q_pred, wincap, make_preds))
    t2 = Thread(target=predict, args=(learn, q_pred, make_preds))
      
    t1.start()
    t2.start()
    

if __name__ == '__main__':
    main()
