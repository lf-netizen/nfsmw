import os
import cv2
import pickle
from datetime import datetime
from utils import preprocess_img, read_angle, read_speed


def preprocess(SRC_PATH, TRACK_IDS, DST_PATH, with_kb_input=False):
    for track_id in TRACK_IDS:
        print(f'Starting {track_id}')
        if not os.path.exists(os.path.join(DST_PATH, str(track_id))):
            os.mkdir(os.path.join(DST_PATH, str(track_id)))
        for img_name in os.listdir(os.path.join(SRC_PATH, str(track_id))):
            img = cv2.imread(os.path.join(SRC_PATH, str(track_id), img_name))#, cv2.IMREAD_GRAYSCALE)
            # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            try:
                img = preprocess_img(img)
            except:
                print(track_id, img_name)
                continue
            # time = img_name[:23]
            # speed = read_speed(img_gray)
            # angle = read_angle(img_gray)
            # kb_input=img_name[24:28] if with_kb_input else ''

            # cv2.imwrite(os.path.join(DST_PATH, str(track_id), f'{time}_{speed}_{angle}_{kb_input}_.png'), img)
            cv2.imwrite(os.path.join(DST_PATH, str(track_id), img_name), img)

def angle_diff_appendix(SRC_PATH, TRACK_IDS, DST_PATH):
    d = {}    
    for track_id in TRACK_IDS:
        print(f'Starting {track_id}')
        for img_name in os.listdir(os.path.join(SRC_PATH, str(track_id))):
            img = cv2.imread(os.path.join(SRC_PATH, str(track_id), img_name), cv2.IMREAD_GRAYSCALE)
            d[img_name.split('_')[0]] = read_angle(img)

    pickle_out = open(f'{DST_PATH}/angle_appendix.pkl', 'wb')
    pickle.dump(d, pickle_out)
    pickle_out.close()
    
if __name__ == '__main__':
    SRC_PATH = 'H:/machine learning/NFSMW_v1/images/720p_RGB_newdiv/zip_buffer'
    # TRACK_IDS = [i for i in range(18 , 34)]; TRACK_IDS.remove(26)
    # TRACK_IDS = [f'tr_{i}' for i in range(16, 23)]
    TRACK_IDS = os.listdir(SRC_PATH)
    DST_PATH = 'H:/machine learning/NFSMW_v1/images_uploadable/720p_RGB_newdiv/zip_buffer'
    preprocess(SRC_PATH, TRACK_IDS, DST_PATH, with_kb_input=True)
    

    # SRC_PATH = 'H:/machine learning/NFSMW_v1/images/handmade_480pRGB'
    # TRACK_IDS = list(range(12, 15)) + list(range(31, 43))
    # # TRACK_IDS = [12]
    # DST_PATH = 'H:/machine learning/NFSMW_v1/data_uploadable/angle_appendix'
    # angle_diff_appendix(SRC_PATH, TRACK_IDS, DST_PATH)

