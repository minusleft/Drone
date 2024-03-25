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

def set_velocity(velocity_x, velocity_y, velocity_z, duration):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, #原本是mavutil.mavlink.MAV_FRAME_BODY_NED
        0b0000111111000111,  # type mask (enable velocity components)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not used)
        0, 0)     # yaw, yaw_rate (not used)

    vehicle.send_mavlink(msg)
    vehicle.flush()
    
    #duration >=time.sleep()
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)

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

#####MAIN EXECUTABLE#####

vehicle = connectMyCopter()
originalaltitude = vehicle.location.global_relative_frame.alt

arm_and_takeoff(1)
set_velocity(1, 0, 0, 2)

vehicle.mode = VehicleMode("LAND")
while vehicle.location.global_relative_frame.alt == originalaltitude:
    time.sleep(1)
print("Land!!")