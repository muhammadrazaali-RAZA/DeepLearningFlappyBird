wide_window  = 288
high_window = 512

import random
import sys
import pygame as pg
from itertools import cycle

pg.init()

#####################################

def hitmask_Get(img):
    msk = []
    for z in range(img.get_width()):
        msk.append([])
        for k in range(img.get_height()):
            msk[z].append(bool(img.get_at((z,k))[3]))
    return msk

def interface_Loader():
    imgs, sonds, hit_mask = {}, {}, {}
    interfacePath = ('objects/redbird-upflap.png','objects/redbird-midflap.png','objects/redbird-downflap.png')
    imgs['numbers'] = (pg.image.load('objects/0.png').convert_alpha(),pg.image.load('objects/1.png').convert_alpha(),pg.image.load('objects/2.png').convert_alpha(),pg.image.load('objects/3.png').convert_alpha(),pg.image.load('objects/4.png').convert_alpha(),pg.image.load('objects/5.png').convert_alpha(),pg.image.load('objects/6.png').convert_alpha(),pg.image.load('objects/7.png').convert_alpha(),pg.image.load('objects/8.png').convert_alpha(),pg.image.load('objects/9.png').convert_alpha())
    imgs['base'] = pg.image.load('objects/base.png').convert_alpha()
    if 'win' in sys.platform:
        type_audio = '.wav'
    else:
        type_audio = '.ogg'
    sonds['die']    = pg.mixer.Sound('audio/die' + type_audio)
    sonds['hit']    = pg.mixer.Sound('audio/hit' + type_audio)
    sonds['point']  = pg.mixer.Sound('audio/point' + type_audio)
    sonds['swoosh'] = pg.mixer.Sound('audio/swoosh' + type_audio)
    sonds['wing']   = pg.mixer.Sound('audio/wing' + type_audio)
    imgs['background'] = pg.image.load('objects/background-black.png').convert()
    imgs['player'] = (pg.image.load(interfacePath[0]).convert_alpha(),pg.image.load(interfacePath[1]).convert_alpha(),pg.image.load(interfacePath[2]).convert_alpha(),)
    imgs['pipe'] = (pg.transform.rotate(pg.image.load('objects/pipe-green.png').convert_alpha(), 180),pg.image.load('objects/pipe-green.png').convert_alpha(),)
    hit_mask['pipe'] = (hitmask_Get(imgs['pipe'][0]),hitmask_Get(imgs['pipe'][1]),)
    hit_mask['player'] = (hitmask_Get(imgs['player'][0]),hitmask_Get(imgs['player'][1]),hitmask_Get(imgs['player'][2]),)
    return imgs, sonds, hit_mask

#####################################

pg.display.set_caption('FB by AI')

Window_scrn = pg.display.set_mode((wide_window, high_window))
imgs, audios, hit_mask = interface_Loader()
wind_basey = high_window * 0.79

class FB_Run_game:
    def __init__(self):
        self.x_base = self.tot_scored = self.plyr_indx = self.itr_looping = 0
        self.x_plyr = int(wide_window * 0.2)
        self.y_plyr = int((high_window - imgs['player'][0].get_height()) / 2)
        self.shift_base = imgs['base'].get_width() - imgs['background'].get_width()
        created_pipe_1 = randomly_generated_pipe()
        created_pipe_2 = randomly_generated_pipe()
        self.top_pipes = [{'x': wide_window, 'y': created_pipe_1[0]['y']},{'x': wide_window + (wide_window / 2), 'y': created_pipe_2[0]['y']},]
        self.bottom_pipes = [{'x': wide_window, 'y': created_pipe_1[1]['y']},{'x': wide_window + (wide_window / 2), 'y': created_pipe_2[1]['y']},]
        self.y_max_val_plyr=10
        self.y_min_val_plyr=-8
        self.y_acc_plyr=1
        self.flap_acc_plyr=-9
        self.x_val_pipe=-4
        self.y_val_plyt=0 
        self.flapped_plyr = False
    def steps_of_frame(self, actions_Input):
        pg.event.pump()
        re_ward = 0.1
        termnl = False
        if sum(actions_Input) != 1:
            raise ValueError('Multiple input actions!')
        if actions_Input[1] == 1:
            if self.y_plyr > -2 * imgs['player'][0].get_height():
                self.y_val_plyt = self.flap_acc_plyr
                self.flapped_plyr = True
                audios['wing'].play()
        pos_mid_plyr = self.x_plyr + imgs['player'][0].get_width() / 2
        for p in self.top_pipes:
            pos_mid_pipe = p['x'] + imgs['pipe'][0].get_width() / 2
            if pos_mid_pipe <= pos_mid_plyr < pos_mid_pipe + 4:
                self.tot_scored += 1
                audios['point'].play()
                re_ward = 1
        if (self.itr_looping + 1) % 3 == 0:
            self.plyr_indx = next(cycle([0, 1, 2, 1]))
        self.itr_looping = (self.itr_looping + 1) % 30
        self.x_base = -((-self.x_base + 100) % self.shift_base)
        if self.y_val_plyt < self.y_max_val_plyr and not self.flapped_plyr:
            self.y_val_plyt += self.y_acc_plyr
        if self.flapped_plyr:
            self.flapped_plyr = False
        self.y_plyr += min(self.y_val_plyt, wind_basey - self.y_plyr - imgs['player'][0].get_height())
        if self.y_plyr < 0:
            self.y_plyr = 0
        for T_pipe, L_pipe in zip(self.top_pipes, self.bottom_pipes):
            T_pipe['x'] += self.x_val_pipe
            L_pipe['x'] += self.x_val_pipe
        if 0 < self.top_pipes[0]['x'] < 5:
            latest_pipe = randomly_generated_pipe()
            self.top_pipes.append(latest_pipe[0])
            self.bottom_pipes.append(latest_pipe[1])
        if self.top_pipes[0]['x'] < -imgs['pipe'][0].get_width():
            self.top_pipes.pop(0)
            self.bottom_pipes.pop(0)
        is_crash= player_collders({'x': self.x_plyr, 'y': self.y_plyr,'index': self.plyr_indx},self.top_pipes, self.bottom_pipes)
        if is_crash:
            audios['hit'].play()
            audios['die'].play()
            termnl = True
            self.__init__()
            re_ward = -1
        Window_scrn.blit(imgs['background'], (0,0))
        for T_pipe, L_pipe in zip(self.top_pipes, self.bottom_pipes):
            Window_scrn.blit(imgs['pipe'][0], (T_pipe['x'], T_pipe['y']))
            Window_scrn.blit(imgs['pipe'][1], (L_pipe['x'], L_pipe['y']))
        Window_scrn.blit(imgs['base'], (self.x_base, wind_basey))
        Window_scrn.blit(imgs['player'][self.plyr_indx],(self.x_plyr, self.y_plyr))
        image_data = pg.surfarray.array3d(pg.display.get_surface())
        pg.display.update()
        pg.time.Clock().tick(30)
        return image_data, re_ward, termnl

def objects_collsion(rct_1, rct_2, hit_mask_1, hit_mask_2):
    rct = rct_1.clip(rct_2)
    if rct.width == 0 or rct.high == 0:
        return False
    x_axis_1, y_axis_1 = rct.x - rct_1.x, rct.y - rct_1.y
    x_axis_2, y_axis_2 = rct.x - rct_2.x, rct.y - rct_2.y
    for i in range(rct.width):
        for k in range(rct.height):
            if hit_mask_1[x_axis_1+i][y_axis_1+k] and hit_mask_2[x_axis_2+i][y_axis_2+k]:
                return True
    return False

def player_collders(plyr, top_pips, botom_pips):
    pi = plyr['index']
    plyr['w'] = imgs['player'][0].get_width()
    plyr['h'] = imgs['player'][0].get_height()
    if plyr['y'] + plyr['h'] >= wind_basey - 1:
        return True
    else:
        plyr_rct = pg.Rect(plyr['x'], plyr['y'],plyr['w'], plyr['h'])
        for T_pips, L_pips in zip(top_pips, botom_pips):
            T_pips_rect = pg.Rect(T_pips['x'], T_pips['y'], imgs['pipe'][0].get_width(),  imgs['pipe'][0].get_height())
            L_pips_rect = pg.Rect(L_pips['x'], L_pips['y'], imgs['pipe'][0].get_width(),  imgs['pipe'][0].get_height())
            top_col = objects_collsion(plyr_rct, T_pips_rect, hit_mask['player'][pi], hit_mask['pipe'][0])
            botm_col = objects_collsion(plyr_rct, L_pips_rect, hit_mask['player'][pi], hit_mask['pipe'][1])
            if top_col or botm_col:
                return True
    return False

def Window_scrn_score(tot_scored):
    wide_of_scrn = 0
    digit_score = [int(x) for x in list(str(tot_scored))]
    for dgt in digit_score:
        wide_of_scrn += imgs['numbers'][dgt].get_width()
    Xoffset = (wide_window - wide_of_scrn) / 2
    for dgt in digit_score:
        Window_scrn.blit(imgs['numbers'][dgt], (Xoffset, high_window * 0.1))
        Xoffset += imgs['numbers'][dgt].get_width()

def randomly_generated_pipe():
    y_axis_gap = [20, 30, 40, 50, 60, 70, 80, 90]
    y_axis_gap = y_axis_gap[random.randint(0, len(y_axis_gap)-1)]
    y_axis_gap += int(wind_basey * 0.2)
    x_axis_pipe = wide_window + 10
    return [{'x': x_axis_pipe, 'y': y_axis_gap -  imgs['pipe'][0].get_height()},{'x': x_axis_pipe, 'y': y_axis_gap + 100},]