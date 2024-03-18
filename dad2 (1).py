"""
dad2.py
depth estimation + object detection
second version

command line:
>>> python3 trt_yolo1.py --usb 0 --model yolov4-tiny-416
"""

import os
import time
import argparse
import torch #depth
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as  np #depth

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

WINDOW_NAME = 'TrtYOLODemo'

model_type = "MiDaS_small" #depth top
midas =  torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms") 

if model_type ==  "DPT_Large" or  model_type ==  "DPT_Hybrid" :
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform #depth last

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#depth top
        input_batch = transform(frame).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = frame.shape[:2],
                mode = "bicubic", 
                align_corners = False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type = cv2. NORM_MINMAX, dtype = cv2.CV_64F)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)	
        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)  

        distances = depth_map.astype("float")#depth top
        distances = np.around(distances, decimals=1)
        for x in range(479,482):
            for y in range(269,272):
                print(distances[y][x], end = " ")       #depth last              

        cv2.rectangle(depth_map, (440,300), (520,230), (0,255,0), 2) #c
        cv2.rectangle(depth_map, (480,350), (560,270), (255,0,0), 2) #1
        cv2.rectangle(depth_map, (400,350), (480,270), (255,0,0), 2) #2
        cv2.rectangle(depth_map, (400,270), (480,190), (255,0,0), 2) #3
        cv2.rectangle(depth_map, (480,270), (560,190), (255,0,0), 2) #4 
            
        cv2.imshow('Depth Map', depth_map)#depth last
    
        cv2.circle(img,(480, 270), 5, (0, 0, 255), cv2.FILLED)

        toc = time.time()
        delay_time = toc - tic #depth
        cv2.putText(img , 'DELAY: {:.2f}'.format(delay_time) , (10,40), cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (150,50,200), 2)#depth
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        
        

        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis)
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
