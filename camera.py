import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import cv2
from utility import *
from yolo import Darknet
import random
import argparse
import pickle as pkl


import os 
import imutils

from num2words import num2words
from subprocess import call
import subprocess



cmd_beg= 'espeak '
cmd_end= ' aplay /home/koryo/Desktop/FACE2/audio/Text.wav  ' # To play back the stored .wav file and to dump the std errors to /dev/null
cmd_out= '--stdout > /home/koryo/Desktop/FACE2/audio/Text.wav ' # To store the voice file
len_seq=50
last=[""]*len_seq
stepa=0
label=""
def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(description='YOLO v3 Real Time Detection')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()


def prep_image(img, inp_dim):
    """Converting a numpy array of frame into PyTorch tensor"""
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim
cmd_beg= 'espeak '
cmd_end= ' aplay /home/koryo/Desktop/FACE2/audio/Text.wav  ' # To play back the stored .wav file and to dump the std errors to /dev/null
cmd_out= '--stdout > /home/koryo/Desktop/FACE2/audio/Text.wav ' # To store the voice file
len_seq=50
last=[""]*len_seq
stepa=0

def write(x, img):

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    global label 
    label = "{0}".format(classes[cls])
    
   
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
     
   
    return img


l=len(label)
print(l)

if __name__ == "__main__":
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "weight/yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    num_classes = 80
    bbox_attrs = 5 + num_classes

    model = Darknet(cfgfile).to(device)
    model.load_weights(weightsfile)

    model.network_info["height"] = args.reso
    inp_dim = int(model.network_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    model.eval()

    cap = cv2.VideoCapture(-1)
    cap.set(3, 720) # set video widht
    cap.set(4, 480) # set video height

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    
    while cap.isOpened():
        if label == True :
            text = label
            text = text.replace(' ', '_')
        
#Calls the Espeak TTS Engine to read aloud a Text
            call([cmd_beg+cmd_out+text+cmd_end], shell=True)
        
            moviepath = '/home/koryo/Desktop/FACE2/audio/Text.wav'
        
            if ( moviepath not in last):
                print(last, "->",moviepath)
                mxprocess = subprocess.Popen(['aplay', moviepath], stdin=subprocess.PIPE)
                last[stepa%len_seq]=moviepath
        stepa +=1
        
        ret, frame = cap.read()
        if ret:
        
            img, orig_im, dim = prep_image(frame, inp_dim)
            img.to(device)

            output = model(img)
            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)
            
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("color/pallete", "rb"))

            list(map(lambda x: write(x, orig_im), output))

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))

        else:
            break
