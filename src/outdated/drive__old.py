import gc
# import tensorflow as tf
import numpy as np
import cv2
import time
from grabscreen import grab_screen
import keyboard as kb
from keras import backend as K
from utils import preprocess_img, MIN_R, MIN_X, MIN_Y, IMG_SIZE_X, IMG_SIZE_Y, read_speed
from create_dataset import CONVERT_INPUT

import timm
from fastai.vision.all import *
from utils import dls_from_np

THRESHOLD = 0.9

TRACKED_KEYS = ['w', 's', 'a', 'd']

def main():
    dummy_x = np.empty(shape=(1, IMG_SIZE_Y, IMG_SIZE_X, 3), dtype=np.uint8)
    dummy_y = np.empty(shape=(1, ))
    dls = dls_from_np(dummy_x, dummy_y, num_train=1)
    learn = vision_learner(dls, 'convnext_tiny_in22k', pretrained=True, model_dir='')
    learn.load('fastai_convnext_tiny_480pRGB2pack_regression_6ep')

    for i in range(2, 0, -1):
        print(i)
        time.sleep(1)

    while True:
        t_start = time.perf_counter()
        screen = grab_screen(region=(1, 31, 1278, 668))
        # screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
        img = preprocess_img(screen)

        # t_start = time.perf_counter()
        pred, _, _ = learn.predict(img)
        # t_stop = time.perf_counter()
        # print("Czas obliczeń:", "{:.7f}".format(t_stop - t_start))
        
        
        
        # pred *= np.array([1/20, 1/20, 1])
        pred = pred[0]
        print(pred)
        
        # REGRESSION
        if np.abs(pred) < THRESHOLD:
            pred = 0
        print(np.sign(pred).astype(int))
        pred = CONVERT_INPUT[np.sign(pred).astype(int)]

        for key, is_pressed in zip(TRACKED_KEYS[2:], pred):
            if is_pressed:
                kb.press(key)
            else:
                kb.release(key)

        if kb.is_pressed('/'):
            break

        t_stop = time.perf_counter()
        print("Czas obliczeń:", "{:.7f}".format(t_stop - t_start))
    for key in TRACKED_KEYS:
        kb.release(key)
    cv2.destroyAllWindows()

##############################
#     [OLD] TF VERSION       #
##############################
# def main():
#     # model_acc = tf.keras.models.load_model('alex_acc.model')
#     for i in range(2, 0, -1):
#         print(i)
#         time.sleep(1)

#     old_angle = 0
#     while True:
#         # t_start = time.perf_counter()
#         screen = grab_screen(region=(1, 31, 1278, 668))
#         screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
#         # cv2.imshow("cv2screen", screen)
#         # cv2.waitKey(10)
#         speed = read_speed(screen)
#         if speed >= 1000:
#             continue

#         # minimap = screen[MIN_Y-MINd_R:MIN_Y+MIN_R, MIN_X-MIN_R:MIN_X+MIN_R]
#         # this_angle = read_angle(minimap)
#         # print(this_angle - old_angle, end='\r')
#         # old_angle = this_angle

#         img = preprocess_img(screen)
#         pred, _, _ = img = img / 255.0

#         pred, _, _ = # t_start = time.perf_counter()
#         t_stop = time.perf_conter()
        
        
#         # pred
#         # pred *= np.array([1/20, 1/20, 1])

#         pred = CONVERT_INPUT[pred]
#         prediction = result_dir
#         if kb.is_pressed('/'):
#             break

#         # continue
    
#         for key, is_pressed in zip(TRACKED_KEYS[2:], prediction):
#             if is_pressed:
#                 kb.press(key)
#             else:
#                 kb.release(key)

#         # t_stop = time.perf_counter()
#         # print("Czas obliczeń:", "{:.7f}".format(t_stop - t_start))

#     for key in TRACKED_KEYS:
#         kb.release(key)
#     cv2.destroyAllWindows()




if __name__ == '__main__':
    main()