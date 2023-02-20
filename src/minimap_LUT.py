import time
import keyboard as kb
import cv2
import os
import numpy as np

from utils import read_speed
from window_capture import WindowCapture

MIN_Y, MIN_X = (569, 128); MIN_R = 16
DIR_NAME = 'minimap_LUT_v2'
RANGE = len(os.listdir(f'images/tests/{DIR_NAME}'))
data  = []
for img_name in os.listdir(f'images/tests/{DIR_NAME}'):
    img = cv2.imread(f'images/tests/{DIR_NAME}/{img_name}')
    data.append(img)

try:
    wincap = WindowCapture('Need for Speed™ Most Wanted')
except:
    wincap = None
false_ctr = 0
prev_val = 0

def get_minimap(screen=None, with_speed=False):
    if screen is None:
        try:
            screen = wincap.get_screenshot()
        except:
            return

    minimap = screen[MIN_Y-MIN_R:MIN_Y+MIN_R, MIN_X-MIN_R:MIN_X+MIN_R]
    minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_RGB2HSV)
    minimap = cv2.bitwise_and(minimap, minimap, mask=(minimap_hsv[:, :, 1] > 130).astype(np.uint8))
    minimap = cv2.bitwise_and(minimap, minimap, mask=(minimap[:, :, 2] > 100).astype(np.uint8))
    minimap = (minimap > 0) * 255
    
    if not with_speed:
        return minimap

    screenshot_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    speed = read_speed(screenshot_gray)
    
    return minimap, speed    

def normalize_it(it):
    if it < 0:
        return it + RANGE
    if it >= RANGE:
        return it - RANGE
    return it

def create():
    # kb.press('a')
    # kb.release('s')
    
    t_start = time.perf_counter()
    minimap, speed = get_minimap(with_speed=True)

    if speed > 12:
        kb.release('w')
    elif speed < 12:
        kb.press('w')

    global data
    repeat = any(np.array_equal(minimap, x) for x in data)
    t_stop = time.perf_counter()

    print(f"Time: {(t_stop - t_start)*1000:.4f}; {repeat}") 

    if repeat:
        return

    cv2.imwrite(f'images/tests/minimap_LUT/{len(data)}.png', minimap)
    data.append(minimap)

def test():
    t_start = time.perf_counter()
    minimap = get_minimap()

    global data
    global prev_val
    # for it, x in enumerate(data):
    #     if np.array_equal(minimap, x):
    #         data_it = it
    #         repeat = True
    #         break
    # else:
    #     global false_ctr
    #     false_ctr += 1
    #     repeat = False
    
    for offset in range(RANGE//2):
        it_forward = normalize_it(prev_val + offset)
        it_backward = normalize_it(prev_val - offset)
        
        if np.array_equal(minimap, data[it_forward]):
            data_it = it_forward
            repeat = True
            break
        if np.array_equal(minimap, data[it_backward]):
            data_it = it_backward
            repeat = True
            break
    else:
        global false_ctr
        false_ctr += 1
        repeat = False

    t_stop = time.perf_counter()
    print(f"Time: {(t_stop - t_start)*1000:.4f}; {repeat} {false_ctr if not repeat else data_it:4} {10000 if not repeat else (prev_val - data_it): 3d}") 

    if not repeat:
        return

    # FOR TESTING MONOTONICITY
    # if prev_val < data_it:
    #     print('--------------------')
    # print(prev_val - data_it)
    
    # time.sleep(0.2 - (t_stop - t_start))
    prev_val = data_it

def read(minimap, prev_val=0):
    for offset in range(RANGE//2):
        it_forward = normalize_it(prev_val + offset)
        it_backward = normalize_it(prev_val - offset)
        
        if np.array_equal(minimap, data[it_forward]):
            data_it = it_forward
            repeat = True
            break
        if np.array_equal(minimap, data[it_backward]):
            data_it = it_backward
            repeat = True
            break
    else:
        return None
    return data_it

def main():
    # start delay
    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    paused = False
    # kb.press('a')
    while True:
        if kb.is_pressed('/'):
            break
        if kb.is_pressed('p'):
            paused = True
        if kb.is_pressed('o'):
            paused = False
            print('      ', end='\r')
        if paused:
            print('Paused', end='\r')
            continue

        # create()
        test()

    kb.release('a')
    kb.release('w')
    kb.release('s')

if __name__ == '__main__':
    main()