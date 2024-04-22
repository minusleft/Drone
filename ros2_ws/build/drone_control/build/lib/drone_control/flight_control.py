import rclpy
from rclpy.node import Node
from pymavlink import mavutil

class FlightControlNode(Node):
    def __init__(self):
        super().__init__('flight_control_node')
        self.connection = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
        self.get_logger().info('Connected to the drone.')

    def arm_and_takeoff(self, target_height):
        while not self.connection.wait_heartbeat(timeout=3):
            self.get_logger().info('Waiting for heartbeat...')
        
        # Arm the drone
        self.connection.mav.command_long_send(
            self.connection.target_system, self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0)  # 1 to arm
        self.get_logger().info('Drone armed')

        # Take off
        self.connection.mav.command_long_send(
            self.connection.target_system, self.connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, target_height)
        self.get_logger().info('Takeoff command sent')

def main(args=None):
    rclpy.init(args=args)
    flight_control = FlightControlNode()
    flight_control.arm_and_takeoff(10)  # Example: target height of 10 meters
    rclpy.spin(flight_control)
    flight_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
