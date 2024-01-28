#####DEPENDENCIES#####

from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import time
import socket
import builtins
import math
import argparse
 
#####FUNCTIONS#####

def connectMyCopter():
 
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect', default='127.0.0.1:14550')
    args = parser.parse_args()
    connection_string = args.connect
 
    vehicle = connect(connection_string, wait_ready=True,timeout=60)
 
    return vehicle
 
def arm_and_takeoff_land(targetHeight):

    if vehicle is None:
        print("No vehicle connected, cannot takeoff.")
        return

    vehicle.mode = VehicleMode('STABILIZE')
    vehicle.arm()
    vehicle.mode = VehicleMode('GUIDED')
    x = vehicle.location.global_relative_frame.alt
    time.sleep(2)
    
    vehicle.simple_takeoff(targetHeight)
    print('Taking off')

    while vehicle.location.global_relative_frame.alt < targetHeight * 0.95:
        time.sleep(1)
    print("Target altitude reached!!")

    vehicle.mode = VehicleMode("LAND")
    while vehicle.location.global_relative_frame.alt == x:
        time.sleep(1)

    print("Land!!")

    return None

#####MAIN EXECUTABLE#####

vehicle = connectMyCopter()
arm_and_takeoff_land(1)