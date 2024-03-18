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

#####MAIN EXECUTABLE#####

vehicle = connectMyCopter()
print('Successful!')
