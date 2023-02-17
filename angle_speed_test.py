import keyboard as kb
import time
from window_capture import WindowCapture

from utils import no_key, read_angle

wincap = WindowCapture('Need for Speed™ Most Wanted')

was_released = False
print('ready')
while True:
    if kb.is_pressed('q'):
        t_start = time.perf_counter()
        prev_angle = 0
        kb.press('w')

        kb.press('a')
        while True:
            if kb.is_pressed('/'):
                no_key()
                raise ValueError()           
            try:
                screenshot = wincap.get_screenshot()
            except:
                continue
            
            minimap = screenshot[553:585, 112:144]
            angle = read_angle(minimap, prev_val=prev_angle)

            diff_time = time.perf_counter() - t_start
            print(f'{diff_time*1000:.5f}', angle)

            if angle is not None:
                prev_angle = angle
            
            if diff_time > 1:
                print('RELEASED a')
                break     
        kb.release('a')
        
        kb.press('d')
        while True:
            if kb.is_pressed('/'):
                no_key()
                raise ValueError()           
            try:
                screenshot = wincap.get_screenshot()
            except:
                continue
            
            minimap = screenshot[553:585, 112:144]
            angle = read_angle(minimap, prev_val=prev_angle)

            diff_time = time.perf_counter() - t_start
            print(f'{diff_time*1000:.5f}', angle)

            if angle is not None:
                prev_angle = angle
            
            if diff_time > 2:
                print('RELEASED d')
                break
        kb.release('d')

        while True:
            if kb.is_pressed('/'):
                no_key()
                raise ValueError()           
            try:
                screenshot = wincap.get_screenshot()
            except:
                continue
            
            minimap = screenshot[553:585, 112:144]
            angle = read_angle(minimap, prev_val=prev_angle)

            diff_time = time.perf_counter() - t_start
            print(f'{diff_time*1000:.5f}', angle)

            if angle is not None:
                prev_angle = angle
            
            if diff_time > 4:
                break   
        print('finished')
        no_key()
        break
        