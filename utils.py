import numpy as np
import cv2
from scipy.signal import find_peaks
from grabscreen import grab_screen
from fastai.vision.all import *
import datetime
from window_capture import WindowCapture
import keyboard as kb


IMG_SIZE_X = int(960 / 2)
IMG_SIZE_Y = int(540 / 2)
def preprocess_img(img):
    img = cv2.resize(img, (IMG_SIZE_X, IMG_SIZE_Y))
    # img = tf.keras.utils.normalize(img)
    # img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img


# FHD VERSION
# MIN_Y, MIN_X = (853, 191); MIN_R = 20
# HD VERSION
MIN_Y, MIN_X = (528, 127); MIN_R = 15

def read_angle__old(img):
    # if len(img.shape) > 2:
    #     img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    minimap = img[MIN_Y-MIN_R:MIN_Y+MIN_R, MIN_X-MIN_R:MIN_X+MIN_R]
    minimap = (minimap > 50).astype(np.uint8)
    cv2.floodFill(minimap, mask=None, seedPoint=(MIN_R, MIN_R), newVal=255)

    minimap = cv2.linearPolar(minimap, (MIN_R, MIN_R), 20, cv2.WARP_FILL_OUTLIERS)

    end_of_line = -np.argmin(minimap, axis=1)
    end_of_line = np.append(end_of_line, -minimap.shape[0])
    # end_of_line = np.append(end_of_line, end_of_line[:10])
    # end_of_line = np.array([sum(np.take(end_of_line, range(i-5, i+6), mode='wrap')) for i in range(end_of_line.shape[0])])
    peaks_idx = find_peaks(end_of_line)[0]
    peaks_val =  -np.take(end_of_line, peaks_idx)
    try:
        angle = peaks_idx[np.argmax(peaks_val)]
        return angle
    except:
        return 0


# FHD VERSION
# MIN_Y, MIN_X = (853, 191); MIN_R = 20
# HD VERSION
# MIN_Y, MIN_X = (528, 127); MIN_R = 15
# 960x540p VERSION
# MIN_Y, MIN_X = (427, 96); MIN_R = 10
# 1280x720 VERSION
# MIN_Y, MIN_X = (569, 128); MIN_R = 16
Line = namedtuple('Line', 'x1 y1 x2 y2 angle')

def read_angle(screen):
    if screen.shape[0] == 720:
        MIN_Y, MIN_X = (569, 128); MIN_R = 16
    elif screen.shape[0] == 668:
        MIN_Y, MIN_X = (528, 127); MIN_R = 15
        
    # get and preprocess minimap
    minimap = screen[MIN_Y-MIN_R:MIN_Y+MIN_R, MIN_X-MIN_R:MIN_X+MIN_R]
    minimap = (minimap > 60).astype(np.uint8)
    cv2.floodFill(minimap, mask=None, seedPoint=(MIN_R, MIN_R), newVal=255)
    
    # find edges and lines corresponding to them
    edges = cv2.Canny(minimap, 0, 150)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/360, 12, minLineLength=8, maxLineGap=10)
    # TESTING
    qwe = False
    cont = False
    thr = 12
    for _ in range(5):
        lines = cv2.HoughLinesP(edges, 1, np.pi/360, thr, minLineLength=8, maxLineGap=5)
        if lines is None or len(lines) < 2:
            thr -= 1
        elif len(lines) == 2:
            cont = True
            break
        elif len(lines) == 3:
            lines_temp = [Line(x1=x1, y1=y1, x2=x2, y2=y2, angle=-np.arctan2((y2-y1), (x2-x1))*180/np.pi) for tpl in lines for x1,y1,x2,y2 in tpl]
            a0 = np.abs(np.abs(lines_temp[1].angle - lines_temp[2].angle) - 40)
            a1 = np.abs(np.abs(lines_temp[0].angle - lines_temp[2].angle) - 40)
            a2 = np.abs(np.abs(lines_temp[0].angle - lines_temp[1].angle) - 40)
            qwe = True
        elif len(lines) > 2:
            thr += 1
    if not cont:
        # print('not found method')
        return

    if qwe:
        del lines[np.argmin([a0, a1, a2])]

    if lines is None:
        return
    # END TESTING
    lines = [Line(x1=x1, y1=y1, x2=x2, y2=y2, angle=-np.arctan2((y2-y1), (x2-x1))*180/np.pi) for tpl in lines for x1,y1,x2,y2 in tpl]
    
    # delete third line corresponding to base (if exists)
    # if len(lines) == 3:
    #     a0 = np.abs(np.abs(lines[1].angle - lines[2].angle) - 40)
    #     a1 = np.abs(np.abs(lines[0].angle - lines[2].angle) - 40)
    #     a2 = np.abs(np.abs(lines[0].angle - lines[1].angle) - 40)
    #     del lines[np.argmin([a0, a1, a2])]

    if len(lines) != 2:
        return None
    
    # find intersection point - from general equaiton ax + by + c = 0
    a1, a2 = ((line.y2-line.y1) / (line.x2-line.x1) if (line.x2-line.x1) != 0 else 1000 for line in lines)
    b1, b2 = -1, -1
    c1, c2 = lines[0].y1 - a1 * lines[0].x1, lines[1].y1 - a2 * lines[1].x1

    if (a1*b2 - a2*b1) == 0:
        return None

    x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
    y = (c1*a2 - c2*a1) / (a1*b2 - a2*b1)

    if not (0 < y < minimap.shape[0] and 0 < x < minimap.shape[1]):
        return

    angles = np.array([np.arctan2(-(line.y1-y), line.x1-x) if (line.x1-x)**2 + (line.y1-y)**2 > (line.x2-x)**2 + (line.y2-y)**2 else np.arctan2(-(line.y2-y), line.x2-x) for line in lines]) * 180/np.pi
    if np.abs(angles[0] - angles[1]) > 180:
        angles[np.argmin(angles)] += 360

    result_angle = np.mean(angles)

    if result_angle > 180:
        result_angle -= 360

    return result_angle

def angle_diff_norm(this_angle, prev_angle):
    if np.abs(this_angle - prev_angle) > 180:
        if this_angle < 0:
            this_angle += 360
        else:
            prev_angle += 360
    diff = this_angle - prev_angle
    if abs(diff) > 30:
        return 0
    return diff


# from IPython.display import clear_output
# while True:
# # for _ in range(1):
#     clear_output(wait=True)
#     screen = grab_screen(region=(0, 31, 1920, 1080))
#     angle = read_angle(screen)
#     print(angle, end='\r')



# Rotating minimap version
# def read_angle(screen):
#     minimap = screen[MIN_Y-MIN_R:MIN_Y+MIN_R, MIN_X-MIN_R:MIN_X+MIN_R]
#     mask = np.zeros(minimap.shape, dtype="uint8")
#     mask = cv2.circle(mask, (MIN_R,MIN_R), int(0.7*MIN_R), (255,255,255), -1)
#     mask_vehicle = np.zeros(minimap.shape, dtype="uint8")
#     mask_vehicle = cv2.circle(mask_vehicle, (MIN_R,MIN_R), 7, (255,255,255), -1)
#     mask = cv2.subtract(mask, mask_vehicle)

#     minimap = cv2.bitwise_and(minimap, minimap, mask=mask)

#     polar = cv2.linearPolar(minimap, (MIN_R, MIN_R), int(0.7*MIN_R), cv2.WARP_FILL_OUTLIERS)
#     polar = polar >= 150


#     return np.average(polar.sum(axis=1), weights=[i for i in range(polar.shape[0])])


AREAS = [
    (1, 8),
    (6, 1),
    (6, 15),
    (11, 8),
    (18, 1),
    (18, 15),
    (23, 8),
]
def get_digit(screen): # 25x17
    display = np.zeros(7, dtype=bool)
    for it, (x, y) in enumerate(AREAS):
        if np.any(screen[x-1:x+2, y-1:y+2]):
            display[it] = True
 
    if   np.all(display == np.array([False, False, False, False, False, False, False])):
        return 0
    elif np.all(display == np.array([True,  True,  True,  False, True,  True,  True ])):
        return 0
    elif np.all(display == np.array([False, False, True,  False, False, True,  False])):
        return 1
    elif np.all(display == np.array([True,  False, True,  True,  True,  False, True ])):
        return 2
    elif np.all(display == np.array([True,  False, True,  True,  False, True,  True ])):
        return 3
    elif np.all(display == np.array([False, True,  True,  True,  False, True,  False])):
        return 4
    elif np.all(display == np.array([True,  True,  False, True,  False, True,  True ])):
        return 5
    elif np.all(display == np.array([True,  True,  False, True,  True,  True,  True ])):
        return 6
    elif np.all(display == np.array([True,  False, True,  False, False, True,  False])):
        return 7
    elif np.all(display == np.array([True,  True,  True,  True,  True,  True,  True ])):
        return 8
    elif np.all(display == np.array([True,  True,  True,  True,  False, True,  True ])):
        return 9
    else:
        return 1000

# FOR 1277..
# def read_speed(screen):
#     screen = screen[555:580, 1117:1185]
#     screen = screen < 40

#     dig1 = screen[:, :17]
#     dig2 = screen[:, 25:25+17]
#     dig3 = screen[:, 51:51+17]

#     return 100*get_digit(dig1) + 10*get_digit(dig2) + 1*get_digit(dig3)

# FOR 1280x720
def read_speed(screen):
    screen = screen[598:625, 1119:1187]
    screen = screen < 40

    dig1 = screen[:, :18]
    dig2 = screen[:, 26:26+18]
    dig3 = screen[:, 50:50+18]

    return 100*get_digit(dig1) + 10*get_digit(dig2) + 1*get_digit(dig3)


def old_way_of_loading_models(): 
    # based on preexissting architectures
    def dls_from_np(images, labels, num_train):
        num_images = images.shape[0]
        
        def pass_index(idx):
            return idx

        def get_x(i):
            return images[i]

        def get_y(i):
            # FOR MINIMAP ROTATION REGRESSION
            return labels[i]
            
            # FOR MINIMAP ROTATION CLASSIFICATION
            # return 0 if np.abs(labels[i]) <= 1 else np.sign(labels[i])
            
            # FOR KB INPUT CLASSIFICATION 
            # return CONVERT_INPUT.index(labels[i][2:])

        dblock = DataBlock(
            blocks=(ImageBlock, RegressionBlock),
            # blocks=(ImageBlock, CategoryBlock),
            get_items=pass_index,
            get_x=get_x,
            get_y=get_y,
            # item_tfms=[Resize((224, 224), method='squish')],
            splitter=IndexSplitter(list(range(num_train, num_images)))
            # splitter=EndSplitter(valid_pct=0.1)
            )
        dls = dblock.dataloaders(list(range(num_images)), shuffle=True, bs=64)
        return dls

    dummy_x = np.empty(shape=(1, IMG_SIZE_Y, IMG_SIZE_X, 3), dtype=np.uint8)
    dummy_y = np.empty(shape=(1, ))
    dls = dls_from_np(dummy_x, dummy_y, num_train=1)
    # learn = vision_learner(dls, 'convnext_tiny_in22k', pretrained=True, model_dir='models')
    # learn.load('convnext_tiny_480pRGB2pack_regression_6ep')

    def speed_loss(inp, speed, angle): return F.mse_loss(inp[:, 0], speed)
    def angle_loss(inp, speed, angle): return F.mse_loss(inp[:, 1], angle)
    def combine_loss(inp, speed, angle): return speed_loss(inp, speed, angle) + angle_loss(inp, speed, angle)
    # learn = vision_learner(dls, 'convnext_tiny_in22k', pretrained=True, loss_func=combine_loss, metrics=(speed_loss, angle_loss), n_out=2, cbs=GradientAccumulation(64))
    learn = vision_learner(dls, 'convnext_tiny_384_in22ft1k', pretrained=True, loss_func=combine_loss, metrics=(speed_loss, angle_loss), n_out=2, cbs=GradientAccumulation(64), model_dir='')
    # learn.load('convnext_small_5ep_deafultarch-logp10')
    # learn.load('convnext_small_2ep_deafultarch-logp10')
    # learn.load('convnext_tiny_384')
    # learn.load('convnext_tiny_384_newangle27k_01off')
    # learn.load('convnext_tiny_384_75k_03off')


def get_learner(model_name):
    # dataloaders
    dummy_x = np.empty(shape=(1, IMG_SIZE_Y, IMG_SIZE_X, 3), dtype=np.uint8)
    dummy_speed = np.empty(shape=(1, ), dtype=np.float32)
    dummy_y = np.empty(shape=(1, ))

    def pass_index(idx):
        return idx
    def get_x(i):
        return dummy_x[i], dummy_speed[i]
    def get_y(i):
        return dummy_y[i]

    dblock = DataBlock(
        blocks=(ImageBlock, RegressionBlock),
        get_items=pass_index,
        get_x=get_x,
        get_y=get_y,
        splitter=IndexSplitter(list(range(1, 1)))
        )
    dls = dblock.dataloaders(list(range(1)), shuffle=True, bs=64)

    # model class
    class ConvnextWithSpeed(nn.Module):
        def __init__(self): 
            super().__init__()
            model = create_timm_model('convnext_tiny_384_in22ft1k', n_out=10)[0]
            head_layers = list(model[-1].children())
            head_layers[4] = torch.nn.Linear(in_features=1537, out_features=512, bias=False)
            self.img_body = nn.Sequential(model[:-1], nn.Sequential(*head_layers[:4]))
            nf = num_features_model(self.img_body)
            self.head = create_head(nf+1, 10, pool=False, concat_pool=False)
            

        # def forward(self, *x):
        #     img, speed = x
        #     x_img = self.img_body(img)
        #     x = torch.cat([x_img, speed[:, None]], dim=1)
        #     return self.head(x)

        def forward(self, x):
            img, speed = x
            x_img = self.img_body(img/255)
            x = torch.cat([x_img, speed[:, None]], dim=1)
            return self.head(x)
        
    def freeze(self):
        self.model.img_body[0].requires_grad_(False)
        for module in self.model.img_body[0].modules():
            if isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(True)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(True)
                module.eval()
        self.freeze_to(-1)
        
    def unfreeze(self):
        self.model.img_body[0].requires_grad_(True)
        self.freeze_to(0)

    # loss functions
    def speed_loss(pred, speed, angle, ws, ad): return F.mse_loss(pred[:, 0], speed)
    def angle_loss(pred, speed, angle, ws, ad): return F.mse_loss(pred[:, 1], angle); print(pred[:, 1], angle);
    def kb_input_loss(pred, speed, angle, ws, ad):
        loss_ws = F.cross_entropy(pred[:, 2:6], ws, ignore_index=0) 
        loss_ad = F.cross_entropy(pred[:, 6:], ad, ignore_index=0)
        return 0 if torch.isnan(loss_ws+loss_ad) else loss_ws + loss_ad
    def combine_loss(pred, speed, angle, ws, ad, reduction='none'): return speed_loss(pred, speed, angle, ws, ad) \
                                                                    + angle_loss(pred, speed, angle, ws, ad) \
                                                                    + kb_input_loss(pred, speed, angle, ws, ad)
    def ws_error(pred, speed, angle, ws, ad): 
        mask = [cat.item()!=0 for cat in ws]
        return 0 if torch.isnan(error_rate(pred[mask, 2:6], ws[mask])) else error_rate(pred[mask, 2:6], ws[mask])
    def ad_error(pred, speed, angle, ws, ad): 
        mask = [cat.item()!=0 for cat in ad]
        return 0 if torch.isnan(error_rate(pred[mask, 6:], ad[mask])) else error_rate(pred[mask, 6:], ad[mask])

    # create model
    learn = Learner(dls, ConvnextWithSpeed(), loss_func=combine_loss, metrics=(speed_loss, angle_loss, kb_input_loss, ws_error, ad_error), cbs=GradientAccumulation(64), model_dir='')
    learn.freeze = types.MethodType(freeze, learn)
    learn.unfreeze = types.MethodType(unfreeze, learn)
    learn.load(model_name)
    return learn

TRACKED_KEYS = ['up', 'down', 'left', 'right']
def save_screen(dir, screen, speed, angle, kb_input):
    if angle is None:
        angle = 'None'
    else:
        angle = f'{angle:.3f}'

    if speed > 1000:
        return

    if kb_input not in ['1000', '1010', '1001', '0010', '0001', '0100', '0110', '0101']:
        return

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

    cv2.imwrite(dir + f'/{time}_{speed}_{angle}_{kb_input}_.png', screen)

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
    
#######################################################
def get_learner_(model_name):
    # dataloaders
    dummy_x = np.empty(shape=(1, IMG_SIZE_Y, IMG_SIZE_X, 3), dtype=np.uint8)
    dummy_speed = np.empty(shape=(1, ), dtype=np.float32)
    dummy_y = np.empty(shape=(1, ))

    def pass_index(idx):
        return idx
    def get_x(i):
        return dummy_x[i], dummy_speed[i]
    def get_y(i):
        return dummy_y[i]

    dblock = DataBlock(
        blocks=(ImageBlock, RegressionBlock),
        get_items=pass_index,
        get_x=get_x,
        get_y=get_y,
        splitter=IndexSplitter(list(range(1, 1)))
        )
    dls = dblock.dataloaders(list(range(1)), shuffle=True, bs=64)

    # model class
    class ConvnextWithSpeed(nn.Module):
        def __init__(self, model): 
            super().__init__()
            head_layers = list(model[-1].children())
            head_layers[4] = torch.nn.Linear(in_features=1537, out_features=512, bias=False)
            self.img_body = nn.Sequential(model[:-1], nn.Sequential(*head_layers[:4]))
            self.head = nn.Sequential(*head_layers[4:])

        def forward(self, x):
            img, speed = x
            x_img = self.img_body(img/255)
            x = torch.cat([x_img, speed[:, None]], dim=1)
            return self.head(x)
    
    # loss functions
    def speed_loss(pred, speed, angle, ws, ad): return F.mse_loss(pred[:, 0], speed)
    def angle_loss(pred, speed, angle, ws, ad): return F.mse_loss(pred[:, 1], angle)
    def kb_input_loss(pred, speed, angle, ws, ad):
        return F.cross_entropy(pred[:, 2:6], ws, ignore_index=-2) + F.cross_entropy(pred[:, 6:], ad, ignore_index=-2)
    def combine_loss(pred, speed, angle, ws, ad, reduction='none'): return speed_loss(pred, speed, angle, ws, ad) \
                                                                        + angle_loss(pred, speed, angle, ws, ad) \
                                                                        + kb_input_loss(pred, speed, angle, ws, ad)
    # create model
    model = ConvnextWithSpeed(create_timm_model('convnext_tiny_384_in22ft1k', n_out=10)[0])
    learn = Learner(dls, model, loss_func=combine_loss, metrics=(speed_loss, angle_loss, kb_input_loss), cbs=GradientAccumulation(64))
    # freeze the model
    learn.model.img_body[0].requires_grad_(False)
    for module in learn.model.img_body[0].modules():
        if isinstance(module, nn.LayerNorm):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            module.eval()
    # load params
    learn.load(model_name)
    return learn

def angle_diff_norm__old(angle_diff):
    if angle_diff > 14:
        angle_diff -= 29
    elif angle_diff < -14:
        angle_diff += 29
        
    if np.abs(angle_diff) > 5:
        return 0
    return angle_diff
