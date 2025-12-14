#!/usr/bin/env python3
"""
ROS 2 Node: Coordinate Converter
Wraps homography.py functionality with ROS 2 interface

Provides Service:
    /maze/convert_coordinates (maze_solver_msgs/ConvertCoordinates)

Publishes:
    /maze/conversion_status (std_msgs/String): Current status
    /maze/waypoints_dobot (std_msgs/Float32MultiArray): Converted waypoints

Subscribes:
    /maze/waypoints_pixel (std_msgs/Float32MultiArray): Input waypoints
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

# Import homography functions from your original file
def homography_from_4pt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute H (3x3) s.t. [X,Y,1]^T ~ H @ [x,y,1]^T using DLT (SVD)."""
    A = []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([-x, -y, -1,  0,  0,  0, x*X, y*X, X])
        A.append([ 0,  0,  0, -x, -y, -1, x*Y, y*Y, Y])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    n = pts.shape[0]
    homo = np.hstack([pts, np.ones((n, 1))])      # Nx3
    mapped = (H @ homo.T).T                        # Nx3
    w = mapped[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return mapped[:, :2] / w


def read_pixels_csv(csv_path: Path) -> np.ndarray:
    """Read pixel coordinates from CSV file"""
    xs, ys = [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        field_map = {k.lower().strip(): k for k in (reader.fieldnames or [])}
        if "x" not in field_map or "y" not in field_map:
            raise ValueError("Input CSV must have headers 'x' and 'y'.")
        kx, ky = field_map["x"], field_map["y"]
        for row in reader:
            xs.append(float(row[kx]))
            ys.append(float(row[ky]))
    return np.column_stack([xs, ys])


def write_dobot_csv(csv_path: Path, dobot_pts: np.ndarray) -> None:
    """Write Dobot coordinates to CSV file"""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for X, Y in dobot_pts:
            writer.writerow([f"{X:.6f}", f"{Y:.6f}"])


DOBOT_CORNERS = [
            (190,-180),
            (195, 120),
            (407, 120),
            (425,-165),
        ]


class ConverterNode(Node):
    """ROS 2 Node for converting pixel coordinates to Dobot coordinates"""
    
    def __init__(self):
        super().__init__('converter_node')
        
        # Declare parameters for CSV file paths
        self.declare_parameter('input_csv', 'maze_captures/waypoints_pixel.csv')
        self.declare_parameter('output_csv', 'maze_captures/waypoints_dobot.csv')
        self.declare_parameter('auto_convert_on_startup', False)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/maze/conversion_status', 10)
        self.waypoints_pub = self.create_publisher(Float32MultiArray, '/maze/waypoints_dobot', 10)
        
        # Subscribers
        self.pixel_sub = self.create_subscription(
            Float32MultiArray, '/maze/waypoints_pixel', self.pixel_waypoints_callback, 10
        )
        
        # State
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        
        self.get_logger().info('Converter Node initialized')
        self.get_logger().info(f'Input CSV: {self.get_parameter("input_csv").value}')
        self.get_logger().info(f'Output CSV: {self.get_parameter("output_csv").value}')
        self.get_logger().info(f'Image size: {self.image_width}x{self.image_height}')
        self.publish_status('idle')
        
        # Auto-convert on startup if parameter is set
        if self.get_parameter('auto_convert_on_startup').value:
            self.convert_from_csv()
    
    def publish_status(self, status: str):
        """Publish current status"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Conversion status: {status}')
    
    def pixel_waypoints_callback(self, msg):
        """
        Automatically convert when pixel waypoints are received via topic
        This also saves to CSV file
        """
        self.get_logger().info(f'Received {len(msg.data)//2} pixel waypoints via topic')
        self.publish_status('converting')
        
        # Convert Float32MultiArray to numpy array
        pixel_array = np.array(msg.data).reshape(-1, 2)
        
        # Convert using homography
        dobot_array = self.convert_pixels_to_dobot(
            pixel_array,
            self.image_width,
            self.image_height
        )
        
        if dobot_array is not None:
            # Save to CSV file
            output_csv = Path(self.get_parameter('output_csv').value)
            try:
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                write_dobot_csv(output_csv, dobot_array)
                self.get_logger().info(f'✓ Saved Dobot waypoints to: {output_csv}')
            except Exception as e:
                self.get_logger().error(f'Failed to save CSV: {e}')
            
            # Publish converted waypoints to topic
            self.publish_dobot_waypoints(dobot_array)
            self.publish_status('done')
        else:
            self.publish_status('error')
    
    def convert_from_csv(self) -> bool:
        """
        Read pixel coordinates from CSV, convert, save to CSV, and publish
        """
        input_csv = Path(self.get_parameter('input_csv').value)
        output_csv = Path(self.get_parameter('output_csv').value)
        
        if not input_csv.exists():
            self.get_logger().error(f'Input CSV not found: {input_csv}')
            self.publish_status('error')
            return False
        
        self.get_logger().info(f'Reading pixel waypoints from: {input_csv}')
        self.publish_status('converting')
        
        try:
            # Read pixel coordinates from CSV
            pixel_array = read_pixels_csv(input_csv)
            self.get_logger().info(f'Loaded {len(pixel_array)} pixel waypoints from CSV')
            
            # Convert using homography
            dobot_array = self.convert_pixels_to_dobot(
                pixel_array,
                self.image_width,
                self.image_height
            )
            
            if dobot_array is not None:
                # Save to CSV file
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                write_dobot_csv(output_csv, dobot_array)
                self.get_logger().info(f'✓ Saved Dobot waypoints to: {output_csv}')
                
                # Publish converted waypoints to topic
                self.publish_dobot_waypoints(dobot_array)
                self.publish_status('done')
                return True
            else:
                self.publish_status('error')
                return False
        
        except Exception as e:
            self.get_logger().error(f'CSV conversion failed: {e}')
            self.publish_status('error')
            return False
    
    def convert_pixels_to_dobot(self, pixel_pts: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """
        Convert pixel coordinates to Dobot coordinates using homography
        
        Args:
            pixel_pts: Nx2 array of pixel coordinates
            img_w: Image width
            img_h: Image height
        
        Returns:
            Nx2 array of Dobot coordinates or None on error
        """
        try:
            # Define pixel-space corners (TL, TR, BR, BL)
            px_corners = np.array([
                [0.0,                 0.0],
                [float(img_w - 1),     0.0],
                [float(img_w - 1),     float(img_h - 1)],
                [0.0,                 float(img_h - 1)],
            ], dtype=float)
            
            # Dobot corners
            dobot_corners = np.asarray(DOBOT_CORNERS, dtype=float)
            
            # Compute homography
            H = homography_from_4pt(px_corners, dobot_corners)
            
            # Apply transformation
            dobot_pts = apply_homography(H, pixel_pts)
            
            self.get_logger().info(f'✓ Converted {len(pixel_pts)} points successfully')
            return dobot_pts
        
        except Exception as e:
            self.get_logger().error(f'Conversion failed: {e}')
            return None
    
    def publish_dobot_waypoints(self, dobot_pts: np.ndarray):
        """Publish Dobot waypoints as Float32MultiArray"""
        msg = Float32MultiArray()
        
        flattened = []
        for X, Y in dobot_pts:
            flattened.append(float(X))
            flattened.append(float(Y))
        
        msg.data = flattened
        self.waypoints_pub.publish(msg)
        self.get_logger().info(f'✓ Published {len(dobot_pts)} Dobot waypoints to topic')


def main(args=None):
    rclpy.init(args=args)
    node = ConverterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()