"""trt_yolo.py
#sudo mavproxy.py --master=/dev/ttyTHS1 --baudrate=57600 --out 127.0.0.1:14550
This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import torch #depth
import numpy as  np #depth

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO


from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import time
import socket
import builtins
import math
import argparse
from pymavlink import mavutil

#####FUNCTIONS#####

def arm_and_takeoff(targetHeight):
    args, vehicle = parse_args()
    if vehicle is None:
        print("No vehicle connected, cannot takeoff.")
        return

    vehicle.mode = VehicleMode('STABILIZE')
    vehicle.arm()
    vehicle.mode = VehicleMode('GUIDED')
    time.sleep(1)

    vehicle.simple_takeoff(targetHeight)
    print('Taking off')

    while vehicle.location.global_relative_frame.alt < targetHeight * 0.95:
        time.sleep(1)
    print("Target altitude reached!!")

def condition_yaw(degrees, relative, direction):
    args, vehicle = parse_args()
    if relative:
        is_relative = 1  # yaw relative to direction of travel
    else:
        is_relative = 0  # yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
        0,  # confirmation
        degrees,  # param 1, yaw in degrees
        0,  # param 2, yaw speed deg/s
        direction,  # param 3, direction -1 ccw, 1 cw
        is_relative,  # param 4, relative offset 1, absolute angle 0
        0, 0, 0)  # param 5 ~ 7 not used
    vehicle.send_mavlink(msg) # send command to vehicle
    vehicle.flush()

def set_velocity(vehicle, velocity_x, velocity_y, velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,  # type mask (enable velocity components)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not used)
        0, 0)     # yaw, yaw_rate (not used)

    vehicle.send_mavlink(msg)
    vehicle.flush()

def forward(vehicle, distance, speed):
    set_velocity(vehicle, speed, 0, 0) #設定飛行速度
    time.sleep(distance / speed) #飛行一段時間
    set_velocity(vehicle, 0, 0, 0) #停止前進
    print("Flying forward complete!")

WINDOW_NAME = 'TrtYOLODemo'
turn = 0
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
    parser.add_argument(
        '--connect', default='127.0.0.1:14550')
    args = parser.parse_args()

    parser = argparse.ArgumentParser(description='commands')
    connection_string = args.connect
 
    vehicle = connect(connection_string, wait_ready=True,timeout=60)

    return args, vehicle

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    args, vehicle = parse_args()
    trun = 0
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break

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
                print(distances[y][x], end = " ") #depth last  

        cv2.rectangle(depth_map, (440,300), (520,230), (0,255,0), 2) #c
        cv2.rectangle(depth_map, (480,350), (560,270), (255,0,0), 2) #1
        cv2.rectangle(depth_map, (400,350), (480,270), (255,0,0), 2) #2
        cv2.rectangle(depth_map, (400,270), (480,190), (255,0,0), 2) #3
        cv2.rectangle(depth_map, (480,270), (560,190), (255,0,0), 2) #4 
        #cv2.imshow('Depth Map', depth_map)#depth last
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img, cls_name, centerx, centery = vis.draw_bboxes(img, boxes, confs, clss)
        if cls_name:
            print(cls_name)
        if cls_name == "person":
            cv2.circle(img,(centerx, centery), 3, (0, 255, 0), cv2.FILLED)
            if centerx > 480 and centery > 270:
                print("down and right")
            if centerx > 480 and centery < 270:
                print("up and right")
            if centerx < 480 and centery > 270:
                print("down and left")
            if centerx < 480 and centery < 270:
                print("up and left")

            horizontal_move_pixel = centerx - 480
            actual_horizontal_angel = horizontal_move_pixel*(90/960)
            print("horizontal_angel = ", actual_horizontal_angel)
            if actual_horizontal_angel > 0:
                trun = 1
            if actual_horizontal_angel < 0:
                trun = -1
            print("1.Relative angle = +30")
            condition_yaw(int(actual_horizontal_angel), 1, trun)
            print("2.Relative angle = +30")

            vertical_move_pixel = 270 - centery
            actual_vertical_angel = vertical_move_pixel*(80/540)
            print("vertical_angel = ", actual_vertical_angel)
            
            move_distances = distances[centery][centerx] - 1
            forward(vehicle, int(move_distances), 1)
        #screen center
        cv2.circle(img,(480, 270), 5, (0, 0, 255), cv2.FILLED)
        #img = show_fps(img, fps)
        #cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        
        delay_time = toc - tic #depth
        cv2.putText(img , 'DELAY: {:.2f}'.format(delay_time) , (10,40), cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (150,50,200), 2)#depth

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
    args, vehicle = parse_args()
    #img, cls_name, centerx, centery = vis.draw_bboxes(img, boxes, confs, clss)
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
    time.sleep(5)

    originalaltitude = vehicle.location.global_relative_frame.alt
    arm_and_takeoff(1)
    loop_and_detect(cam, trt_yolo, args.conf_thresh, vis=vis)
    cam.release()
    cv2.destroyAllWindows()
"""
    obj = name(cls_dict)
    #print(name(cls_dict))
    if obj == "person":
        cv2.circle(img,(centerx, centery), 3, (0, 255, 0), cv2.FILLED)
        if centerx > 480 and centery > 270:
            print("down and right")
        if centerx > 480 and centery < 270:
            print("up and right")
        if centerx < 480 and centery > 270:
            print("down and left")
        if centerx < 480 and centery < 270:
            print("up and left")
        horizontal_move_pixel = centerx - 480
        actual_horizontal_angel = horizontal_move_pixel*(90/960)
        print("horizontal_angel = ", actual_horizontal_angel)
        vertical_move_pixel = 270 - centery
        actual_vertical_angel = vertical_move_pixel*(80/540)
        print("vertical_angel = ", actual_vertical_angel)
"""
    

if __name__ == '__main__':
    main()
