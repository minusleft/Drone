#####DEPENDENCIES#####

from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import time
import socket
import builtins
import math
import argparse
from pymavlink import mavutil

#####FUNCTIONS#####

def connectMyCopter():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect', default='127.0.0.1:14550')
    args = parser.parse_args()
    connection_string = args.connect

    vehicle = connect(connection_string, wait_ready=True,timeout=60)

    return vehicle

def arm_and_takeoff(targetHeight):
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

#####MAIN EXECUTABLE#####

vehicle = connectMyCopter()
originalaltitude = vehicle.location.global_relative_frame.alt

arm_and_takeoff(1)
time.sleep(2)
condition_yaw(10, 1, 1)
print("Relative angle = +10")
time.sleep(2)
condition_yaw(10, 1, -1)
print("Relative angle = -10")
time.sleep(2)
condition_yaw(20, 1, -1)
print("Relative angle = -20")
time.sleep(2)

vehicle.mode = VehicleMode("LAND")
while vehicle.location.global_relative_frame.alt == originalaltitude:
    time.sleep(1)
print("Land!!")