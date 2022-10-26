import numpy as np
import cv2 
import random
import os
import datetime
import pickle
import math
from utils import preprocess_img, IMG_SIZE_X, IMG_SIZE_Y, read_speed, read_angle, angle_diff_norm
from datetime import datetime, timedelta
from collections import namedtuple

# SHIFT_SPEED = 0
CONVERT_INPUT = [[0, 0], [0, 1], [1, 0]]

# # TRACK_IDS = [
#     # i for i in range(34, 45)
#     # 40
#     # 41,
#     # 42,
#     # 43,
#     # 44,
#     # 45,
#     # 46
# # ]
# TRACK_IDS = os.listdir(PATH)
# TRACK_IDS = [46]

TIME_LIMIT = 1.7
TIME_OFFSET = 1

def create_dataset(SRC_PATH, TRACK_IDS, DST_PATH, with_kb_input=False):
    Img = namedtuple('Img', 'content time speed angle kb_input')
    dataset = []
    for track_id in TRACK_IDS:
        print(f'Starting {track_id}')
        queue = []
        PATH_TRACK = SRC_PATH + f'/{track_id}'
        for img_name in os.listdir(PATH_TRACK):
            # READ RAW IMAGE
            img = cv2.imread(os.path.join(PATH_TRACK, img_name))
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # GET PROPERTIES / PREPROCESS
            img = Img(  content=preprocess_img(img), 
                        time=datetime.strptime(img_name[:23], "%Y-%m-%d-%H-%M-%S-%f"),
                        speed=read_speed(img_gray),
                        angle=read_angle(img_gray),
                        kb_input=img_name[24:28] if with_kb_input else None
                        )

            # ADD TO QUEUE AWITING FOR FUTURE ANGLE DATA
            queue.append(img)
            while img.time - queue[0].time > timedelta(seconds=TIME_LIMIT):
                queue.pop(0)
            if img.time - queue[0].time < timedelta(seconds=TIME_OFFSET):
                continue
            
            # GET SPEED/ANGLE DIFF
            img = queue.pop(0)
            speed_diff = queue[-1].speed - img.speed
            future_angle = queue[-1].angle

            angle_diff = angle_diff_norm(future_angle - img.angle)

            # ADD TO DATASET
            dataset.append((img.content, img.speed, speed_diff, angle_diff, img.kb_input))


            # FOR ME-GENERATED IMAGES
            # speed = read_speed(img)

            # img = preprocess_img(img)

            # kb_input = [int(key_pressed) for key_pressed in img_name[-8:-4]]
            # direction_idx = CONVERT_INPUT.index(kb_input[2:])
            # FOR BOT-GENERATED IMAGES
            # speed = int(img_name.split('_')[-2])
            # # rotation = int(os.listdir(PATH_TRACK)[idx+SHIFT_SPEED].split('_')[-3])
            # rotation = int(img_name.split('_')[-3])

            # img = preprocess_img(img)

            # if rotation < 0:
            #     direction_idx = 0
            # elif rotation > 0:
            #     direction_idx = 1
            # elif rotation == 0:
            #     direction_idx = 2



    random.shuffle(dataset)

    x_img, x_speed = [], []
    y_acceleration, y_direction, y_input = [], [], []
    for img, speed, speed_diff, angle_diff, kb_input in dataset:
        # acc = np.zeros(shape=(3,))
        # dir = np.zeros(shape=(3,))

        # acc[] = 1
        # dir[direction_idx] = 1

        x_img.append(img)
        x_speed.append(speed)
        y_acceleration.append(speed_diff)
        y_direction.append(angle_diff)
        if with_kb_input:
            y_input.append(kb_input)
        
    x_img = np.array(x_img).reshape(-1, IMG_SIZE_Y, IMG_SIZE_X, 3)

    if not os.path.exists(DST_PATH):
        os.mkdir(DST_PATH)
    if not os.path.exists(DST_PATH + '/x_img'):
        os.mkdir(DST_PATH + '/x_img')

    for data, name in zip([x_speed, y_direction, y_acceleration],
                          ['x_speed', 'y_direction', 'y_acceleration']):
        data = np.asarray(data)
        pickle_out = open(f'{DST_PATH}/{name}.pickle', 'wb')
        pickle.dump(data, pickle_out)
        pickle_out.close()

    if with_kb_input:
        y_input = np.asarray(y_input)
        pickle_out = open(f'{DST_PATH}/y_input.pickle', 'wb')
        pickle.dump(y_input, pickle_out)
        pickle_out.close()

    SINGLE_INPUT_SIZE = 5000
    NUM_SAMPLES = x_img.shape[0]
    idx_start = [i*SINGLE_INPUT_SIZE for i in range(math.ceil(NUM_SAMPLES / SINGLE_INPUT_SIZE))]
    idx_finish = idx_start[1:] + [NUM_SAMPLES]

    for start, finish in zip(idx_start, idx_finish):
        pickle_out = open(f'{DST_PATH}/x_img/{start}_{finish}_.pickle', 'wb')
        pickle.dump(x_img[start:finish], pickle_out)
        pickle_out.close()
        

    # pickle_out = open('class_weights.pickle', 'wb')
    # pickle.dump(class_weights, pickle_out)
    # pickle_out.close()

    # pickle_out = open('X.pickle', 'wb')
    # pickle.dump(x, pickle_out)
    # pickle_out.close()

    # pickle_out = open('y.pickle', 'wb')
    # pickle.dump(y, pickle_out)
    # pickle_out.close()


if __name__ == '__main__':
    SRC_PATH = 'H:/machine learning/NFSMW_v1/images/handmade_480pRGB'
    DST_PATH = 'H:/machine learning/NFSMW_v1/data_uploadable/handmade_480pRGB/12_14'
    # TRACK_IDS = list(range(12, 16)) # VALIDATION
    # TRACK_IDS = list(range(16, 41)); TRACK_IDS.remove(26) # TRAIN
    # TRACK_IDS = [35]
    # TRACK_IDS = os.listdir(SRC_PATH)
    # track_paths = sorted([f'{SRC_PATH}/{subdir}/{subsubdir}/{track_id}' for subdir in os.listdir(SRC_PATH) for subsubdir in os.listdir(f'{SRC_PATH}/{subdir}') for track_id in os.listdir(f'{SRC_PATH}/{subdir}/{subsubdir}')], key=lambda name: int(name.split('/')[-1]) if '_' not in name.split('/')[-1] else int(name.split('/')[-1].split('_')[0])+1000)
    # ^ one line, 345 signs; for the record
    TRACK_IDS = list(range(12, 15))
    create_dataset(SRC_PATH, TRACK_IDS, DST_PATH, with_kb_input=True)