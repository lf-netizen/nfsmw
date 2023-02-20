import cv2
import numpy as np
import time
# from grabscreen import grab_screen
import keyboard as kb
import datetime
from threading import Event
import os

from utils import read_angle, read_speed, IMG_SIZE_X, IMG_SIZE_Y, preprocess_img
from window_capture import WindowCapture

# DIR = 'images/handmade_480pRGB/480pRGB/29'
DIR = 'images/720p_RGB_newdiv/zip_buffer/LR_spr_45_kb'
if not os.path.exists(DIR):
    os.mkdir(DIR)

TRACKED_KEYS = ['up', 'down', 'left', 'right']

prev_angle = 0
prev_speed = 0
stop_while = False
wincap = WindowCapture('Need for Speed™ Most Wanted')

def capture_screen(with_kb_input=False):
    screen = wincap.get_screenshot()
    
    kb_input = ''
    if with_kb_input:
        for key in TRACKED_KEYS:
            kb_input += str(int(kb.is_pressed(key)))

    if with_kb_input:    
        if kb_input not in ['1000', '0010', '0001', '1010', '1001', '0100', '0110', '0101']:
            return

    now = datetime.datetime.now()
    img_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    
    time = now.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    # img = preprocess_img(screen)
    img = screen
    speed = read_speed(img_gray)
    angle = read_angle(img_gray)

    if angle is None:
        angle = 'None'
    else:
        angle = f'{angle:.3f}'

    if speed > 1000:
        global stop_while
        stop_while = True
        return

    # cv2.imwrite(DIR + f'/{now}_{diff_angle}_{speed}_{diff_speed}.png', screen)
    cv2.imwrite(DIR + f'/{time}_{speed}_{angle}_{kb_input}_.png', img)


def main_bot():
    # start delay
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    paused = False
    while True:
        if kb.is_pressed('/') or stop_while:
            break
        if kb.is_pressed('p'):
            paused = True
        if kb.is_pressed('o'):
            paused = False
        if paused:
            print('Paused', end='\r')
            continue
        capture_screen(with_kb_input=False)
        time.sleep(0.1)
        

def main():
    # start delay
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    ctr = 0
    while True:
        if kb.is_pressed('/') or stop_while:
            break

        if kb.is_pressed('left') or kb.is_pressed('right') or kb.is_pressed('down'):
            capture_screen(with_kb_input=True)
            ctr = 0
        else:
            ctr += 1

        if ctr == 3:
            capture_screen(with_kb_input=True)
            ctr = 0

        time.sleep(0.1)

def main_consistent():
    # start delay
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    paused = False
    while True:
        if kb.is_pressed('/') or stop_while:
            break
        if kb.is_pressed('p'):
            paused = True
        if kb.is_pressed('o'):
            paused = False
            print('      ', end='\r')
        if paused:
            print('Paused', end='\r')
            continue

        capture_screen(with_kb_input=True)
        time.sleep(0.03)
    


if __name__ == '__main__':
    # main_bot()
    # main()
    main_consistent()
    cv2.destroyAllWindows()
