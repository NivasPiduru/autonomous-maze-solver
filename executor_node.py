#!/home/nivas/Downloads/venvs/lab/bin/python3
"""
ROS 2 Node: Dobot Executor with Auto-Home
Wraps test_travel.py functionality with ROS 2 interface

Provides Services:
    /dobot/execute (maze_solver_msgs/ExecutePath)
    /dobot/stop (std_srvs/Trigger)
    /dobot/home (std_srvs/Trigger)

Publishes:
    /dobot/status (std_msgs/String): Current status
    /dobot/progress (std_msgs/Float32): Execution progress (0.0 to 1.0)
    /dobot/current_waypoint (std_msgs/Int32): Current waypoint index

Subscribes:
    /maze/waypoints_dobot (std_msgs/Float32MultiArray): Auto-execute option
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Int32, Float32MultiArray
import numpy as np
import time
import csv
from pathlib import Path

try:
    from pydobot.dobot import MODE_PTP
    import pydobot
    PYDOBOT_AVAILABLE = True
    print("✓ pydobot imported successfully!")
except ImportError as e:
    PYDOBOT_AVAILABLE = False
    print(f"✗ pydobot import failed: {e}")
except Exception as e:
    PYDOBOT_AVAILABLE = False
    print(f"✗ pydobot error: {e}")


# Configuration
DOBOT_PORT = "/dev/ttyACM0"
FIXED_Z = -45.0
FIXED_R = 0.0
SPEED_LIN = 10
SPEED_ANG = 10


def load_waypoints_from_csv(csv_path: Path) -> np.ndarray:
    """
    Load waypoints from CSV file
    
    Args:
        csv_path: Path to CSV file with x,y columns
    
    Returns:
        Nx2 numpy array of waypoints
    """
    waypoints = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                x = float(row[0].strip())
                y = float(row[1].strip())
                waypoints.append([x, y])
            except ValueError:
                continue
    
    if not waypoints:
        raise ValueError("No valid waypoints found in CSV")
    
    return np.array(waypoints)


class ExecutorNode(Node):
    """ROS 2 Node for executing waypoint paths on Dobot"""
    
    def __init__(self):
        super().__init__('executor_node')
        
        # Declare parameters
        self.declare_parameter('waypoints_csv', 'maze_captures/waypoints_dobot.csv')
        self.declare_parameter('auto_execute_on_startup', False)
        self.declare_parameter('z_height', FIXED_Z)
        self.declare_parameter('speed_linear', SPEED_LIN)
        self.declare_parameter('speed_angular', SPEED_ANG)
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/dobot/status', 10)
        self.progress_pub = self.create_publisher(Float32, '/dobot/progress', 10)
        self.waypoint_pub = self.create_publisher(Int32, '/dobot/current_waypoint', 10)
        
        # Subscribers (optional auto-execute from topic)
        self.waypoints_sub = self.create_subscription(
            Float32MultiArray, '/maze/waypoints_dobot', self.waypoints_callback, 10
        )
        
        # State
        self.device = None
        self.connected = False
        self.executing = False
        self.should_stop = False
        self.simulation_mode = False
        
        # Get parameters
        self.z_height = self.get_parameter('z_height').value
        self.speed_linear = self.get_parameter('speed_linear').value
        self.speed_angular = self.get_parameter('speed_angular').value
        
        # Try to connect to Dobot
        if PYDOBOT_AVAILABLE:
            self.connect_dobot()
        else:
            self.get_logger().warn('pydobot not available - running in SIMULATION mode')
            self.simulation_mode = True
        
        self.get_logger().info('Executor Node initialized')
        self.get_logger().info(f'Waypoints CSV: {self.get_parameter("waypoints_csv").value}')
        self.get_logger().info(f'Z height: {self.z_height}')
        
        if self.simulation_mode:
            self.get_logger().info('🎮 SIMULATION MODE: Will print waypoints instead of moving')
        
        self.publish_status('idle')
        
        # Auto-execute on startup if parameter is set
        if self.get_parameter('auto_execute_on_startup').value:
            self.execute_from_csv()
    
    def publish_status(self, status: str):
        """Publish current status"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Dobot status: {status}')
    
    def publish_progress(self, progress: float):
        """Publish execution progress (0.0 to 1.0)"""
        msg = Float32()
        msg.data = float(progress)
        self.progress_pub.publish(msg)
    
    def publish_current_waypoint(self, index: int):
        """Publish current waypoint index"""
        msg = Int32()
        msg.data = index
        self.waypoint_pub.publish(msg)
    
    def connect_dobot(self) -> bool:
        """Connect to Dobot robot"""
        if not PYDOBOT_AVAILABLE:
            self.get_logger().warn('pydobot not available - running in simulation mode')
            self.simulation_mode = True
            return False
        
        try:
            self.device = pydobot.Dobot(port=DOBOT_PORT)
            
            # CRITICAL: Clear any alarms first
            self.get_logger().info('Clearing alarms...')
            self.device.clear_alarms()
            time.sleep(0.5)
            
            # Set speed
            self.device.speed(self.speed_linear, self.speed_angular)
            
            # Get current position to verify connection
            try:
                pose = self.device.pose()
                self.get_logger().info(f'Current position: X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}')
                
                # Check if position is within safe range for Dobot Magician Lite
                if pose[0] < 50 or pose[0] > 700:
                    self.get_logger().warn(f'⚠️  X position {pose[0]:.2f} may be outside safe range (100-500mm)')
                if abs(pose[1]) > 400:
                    self.get_logger().warn(f'⚠️  Y position {pose[1]:.2f} may be outside safe range (-250 to 250mm)')
                    
            except Exception as e:
                self.get_logger().warn(f'Could not read position: {e}')
            
            self.connected = True
            self.simulation_mode = False
            self.get_logger().info(f'✓ Connected to Dobot on {DOBOT_PORT}')
            return True
            
        except Exception as e:
            self.get_logger().warn(f'Could not connect to Dobot: {e}')
            self.get_logger().info('🎮 Switching to SIMULATION mode')
            self.connected = False
            self.simulation_mode = True
            return False
    
    def disconnect_dobot(self):
        """Disconnect from Dobot"""
        if self.device is not None:
            try:
                self.device.close()
                self.get_logger().info('Dobot disconnected')
            except Exception as e:
                self.get_logger().warn(f'Error disconnecting: {e}')
            finally:
                self.device = None
                self.connected = False
    
    def execute_from_csv(self) -> bool:
        """
        Load waypoints from CSV file and execute them
        """
        csv_path = Path(self.get_parameter('waypoints_csv').value)
        
        if not csv_path.exists():
            self.get_logger().error(f'Waypoints CSV not found: {csv_path}')
            self.publish_status('error')
            return False
        
        self.get_logger().info(f'Loading waypoints from: {csv_path}')
        
        try:
            # Load waypoints from CSV
            waypoints = load_waypoints_from_csv(csv_path)
            self.get_logger().info(f'Loaded {len(waypoints)} waypoints from CSV')
            
            # Execute waypoints
            return self.execute_waypoints(waypoints, self.z_height, self.speed_linear)
        
        except Exception as e:
            self.get_logger().error(f'Failed to load or execute waypoints: {e}')
            self.publish_status('error')
            return False
    
    def waypoints_callback(self, msg):
        """
        Auto-execute when waypoints received via topic
        (Alternative to CSV-based execution)
        """
        if self.executing:
            self.get_logger().warn('Already executing, ignoring new waypoints')
            return
        
        # Check data format
        if len(msg.data) % 2 == 0:
            # Data has no Z: [x, y, x, y, ...]
            waypoints = np.array(msg.data).reshape(-1, 2)
        elif len(msg.data) % 3 == 0:
            # Data has Z: [x, y, z, x, y, z, ...]
            data = np.array(msg.data).reshape(-1, 3)
            waypoints = data[:, :2]  # Take only X, Y
        else:
            self.get_logger().error(f'Invalid data length: {len(msg.data)}')
            return
        
        num_waypoints = len(waypoints)
        self.get_logger().info(f'Received {num_waypoints} Dobot waypoints via topic')
        
        # Execute (real or simulation)
        self.execute_waypoints(waypoints, self.z_height, self.speed_linear)
    
    def execute_waypoints(self, waypoints: np.ndarray, z_height: float, speed: float) -> bool:
        """
        Execute waypoint path on Dobot (real or simulation)
        
        Args:
            waypoints: Nx2 array of (X,Y) coordinates
            z_height: Fixed Z height
            speed: Linear speed
        
        Returns:
            True if successful, False otherwise
        """
        if self.executing:
            self.get_logger().warn('Already executing')
            return False
        
        # Safety check: Filter out waypoints outside safe workspace
        if not self.simulation_mode:
            safe_waypoints = []
            for i, (x, y) in enumerate(waypoints):
                # Dobot Magician Lite safe workspace (approximate)
                if 50 <= x <= 700 and -400 <= y <= 400:
                    safe_waypoints.append([x, y])
                else:
                    self.get_logger().warn(f'⚠️  Skipping unsafe waypoint {i}: X={x:.2f}, Y={y:.2f}')
            
            if len(safe_waypoints) == 0:
                self.get_logger().error('❌ All waypoints outside safe workspace! Aborting.')
                self.publish_status('error')
                return False
            
            if len(safe_waypoints) < len(waypoints):
                self.get_logger().warn(f'⚠️  Filtered {len(waypoints) - len(safe_waypoints)} unsafe waypoints')
                waypoints = np.array(safe_waypoints)
        
        self.executing = True
        self.should_stop = False
        self.publish_status('moving')
        self.publish_progress(0.0)
        
        start_time = time.time()
        num_waypoints = len(waypoints)
        completed = 0
        
        # Print header
        if self.simulation_mode:
            self.get_logger().info('=' * 60)
            self.get_logger().info('🎮 SIMULATION MODE - Waypoint Execution')
            self.get_logger().info('=' * 60)
        
        try:
            for i, (x, y) in enumerate(waypoints):
                # Check for stop signal
                if self.should_stop:
                    self.get_logger().warn('Execution stopped by user')
                    break
                
                # Move to waypoint (real or simulated)
                if self.simulation_mode:
                    # SIMULATION MODE: Print coordinates
                    self.get_logger().info(f'[{i+1}/{num_waypoints}] 📍 Moving to: X={x:7.2f}, Y={y:7.2f}, Z={z_height:7.2f}')
                    time.sleep(0.3)  # Simulate movement time
                    completed += 1
                else:
                    # REAL MODE: Actually move
                    self.get_logger().info(f'Moving to waypoint {i+1}/{num_waypoints}: ({x:.2f}, {y:.2f})')
                    
                    try:
                        # Clear alarms before each move
                        self.device.clear_alarms()
                        
                        qid = self.device.move_to(
                            mode=int(MODE_PTP.MOVJ_XYZ),
                            x=float(x),
                            y=float(y),
                            z=z_height,
                            r=FIXED_R
                        )
                        self.device.wait_for_cmd(qid)
                        completed += 1
                        
                    except KeyboardInterrupt:
                        self.get_logger().warn('Interrupted by keyboard')
                        if not self.simulation_mode:
                            self.device.force_stop_and_go()
                        break
                    except Exception as e:
                        self.get_logger().error(f'Error at waypoint {i}: {e}')
                        # Try to clear alarm and continue
                        try:
                            self.get_logger().info('Attempting to clear alarm...')
                            self.device.clear_alarms()
                            time.sleep(0.5)
                        except:
                            pass
                        break
                
                # Publish progress
                progress = (i + 1) / num_waypoints
                self.publish_progress(progress)
                self.publish_current_waypoint(i)
            
            execution_time = time.time() - start_time
            
            if completed == num_waypoints:
                self.publish_status('complete')
                self.publish_progress(1.0)
                
                if self.simulation_mode:
                    self.get_logger().info('=' * 60)
                    self.get_logger().info(f'✓ Simulation complete! {completed}/{num_waypoints} waypoints in {execution_time:.2f}s')
                    self.get_logger().info('=' * 60)
                else:
                    self.get_logger().info(f'✓ Execution complete! {completed}/{num_waypoints} waypoints in {execution_time:.2f}s')
                
                # AUTO-HOME after completing path
                self.get_logger().info('🏠 Returning to home position...')
                time.sleep(0.5)  # Small pause before homing
                home_success = self.go_home()
                if home_success:
                    if self.simulation_mode:
                        self.get_logger().info('✓ [Simulation] Successfully returned to home position')
                    else:
                        self.get_logger().info('✓ Successfully returned to home position')
                    
                    # CRITICAL: Clear alarms and reset state for next task
                    if not self.simulation_mode:
                        try:
                            self.device.clear_alarms()
                            time.sleep(0.2)
                            self.get_logger().info('🔄 Robot ready for next task')
                        except Exception as e:
                            self.get_logger().warn(f'Could not clear final alarms: {e}')
                    
                    self.publish_status('idle')
                else:
                    self.get_logger().warn('⚠ Could not return to home position')
                
                return True
            else:
                self.publish_status('error')
                self.get_logger().error(f'Execution incomplete: {completed}/{num_waypoints}')
                return False
        
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')
            self.publish_status('error')
            return False
        
        finally:
            self.executing = False
    
    def stop_execution(self) -> bool:
        """Emergency stop"""
        if not self.executing:
            self.get_logger().warn('Not executing, nothing to stop')
            return False
        
        self.should_stop = True
        
        if self.connected and self.device is not None and not self.simulation_mode:
            try:
                self.device.force_stop_and_go()
                self.get_logger().info('Emergency stop executed')
            except Exception as e:
                self.get_logger().error(f'Could not stop robot: {e}')
                return False
        else:
            self.get_logger().info('🎮 [Simulation] Emergency stop executed')
        
        self.publish_status('idle')
        return True
    
    def go_home(self) -> bool:
        """Move to home position"""
        home_x, home_y = 240.0, 0.0
        
        if self.simulation_mode:
            # SIMULATION MODE: Print home coordinates
            self.get_logger().info(f'🏠 [Simulation] Moving to home')
            time.sleep(0.3)
            self.get_logger().info('✓ [Simulation] Home position reached')
            return True
        
        if not self.connected or self.device is None:
            self.get_logger().error('Dobot not connected')
            return False
        
        self.get_logger().info(f'Moving to home position ({home_x}, {home_y})...')
        
        try:
            # Clear alarms before homing
            self.device.clear_alarms()
            time.sleep(0.2)
            
            qid = self.device.home()
            self.device.wait_for_cmd(qid)
            
            # CRITICAL: Clear alarms after homing to reset for next task
            time.sleep(0.3)
            self.device.clear_alarms()
            self.get_logger().info('✓ Home position reached and ready for next task')
            
            return True
        except Exception as e:
            self.get_logger().error(f'Could not move home: {e}')
            # Try to clear alarms even if homing failed
            try:
                self.device.clear_alarms()
            except:
                pass
            return False
    
    def reset_robot(self) -> bool:
        """
        Reset robot state - clears alarms and prepares for next task
        Call this if robot becomes unresponsive between tasks
        """
        if self.simulation_mode:
            self.get_logger().info('🔄 [Simulation] Robot reset')
            return True
        
        if not self.connected or self.device is None:
            self.get_logger().error('Dobot not connected')
            return False
        
        self.get_logger().info('🔄 Resetting robot state...')
        
        try:
            # Clear all alarms
            self.device.clear_alarms()
            time.sleep(0.5)
            
            # Clear alarms again (double-clear for stubborn cases)
            self.device.clear_alarms()
            time.sleep(0.3)
            
            # Verify we can read position
            try:
                pose = self.device.pose()
                self.get_logger().info(f'✓ Robot reset complete. Position: X={pose[0]:.2f}, Y={pose[1]:.2f}')
            except:
                self.get_logger().warn('Could not verify position after reset')
            
            self.publish_status('idle')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Reset failed: {e}')
            return False
    
    def __del__(self):
        """Cleanup on node destruction"""
        self.disconnect_dobot()


def main(args=None):
    rclpy.init(args=args)
    node = ExecutorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.disconnect_dobot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()