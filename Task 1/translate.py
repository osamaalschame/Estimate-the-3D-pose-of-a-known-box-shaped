import numpy as np
import open3d as o3d
import cv2

def segment_box(depth_map, color_img, intrinsics, depth_min=2.0, depth_max=3.0):
    """Convert depth map to point cloud, segment box points, and reject outliers."""
    try:
        height, width = depth_map.shape
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Color image shape: {color_img.shape}")
        
        # Check if color image is single-channel (grayscale)
        if len(color_img.shape) == 2 or color_img.shape[-1] == 1:
            print("Converting single-channel color image to 3-channel RGB")
            if len(color_img.shape) == 3 and color_img.shape[-1] == 1:
                color_img = color_img.squeeze(-1)
            color_img = np.stack([color_img] * 3, axis=-1)
            print(f"Converted color image shape: {color_img.shape}")
        
        # Check if color image needs resizing
        if color_img.shape[:2] != depth_map.shape:
            print(f"Resizing color image from {color_img.shape[:2]} to {depth_map.shape}")
            color_img = cv2.resize(color_img, (width, height), interpolation=cv2.INTER_LINEAR)
            print(f"Resized color image shape: {color_img.shape}")
        
        fx, fy = intrinsics[0,0], intrinsics[1,1]
        cx, cy = intrinsics[0,2], intrinsics[1,2]
        
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_map
        
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy
        
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = color_img.reshape(-1, 3) / 255.0
        
        colors = np.clip(colors, 0, 1)
        
        if points.shape[0] != colors.shape[0]:
            raise ValueError(f"Shape mismatch: points ({points.shape[0]}) vs colors ({colors.shape[0]})")
        
        z_flat = z.reshape(-1)
        valid = (z_flat >= depth_min) & (z_flat <= depth_max) & np.isfinite(z_flat)
        
        if valid.shape[0] != points.shape[0]:
            raise ValueError(f"Valid mask shape ({valid.shape[0]}) does not match points ({points.shape[0]})")
        
        points = points[valid]
        colors = colors[valid]
        
        print(f"Points after depth filtering: {len(points)}")
        
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(colors)):
            raise ValueError("NaN or Inf values detected in points or colors")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        print(f"Points after outlier rejection: {len(points)}")
        
        return points, colors
    except Exception as e:
        raise ValueError(f"Error in segment_box: {str(e)}")

def fit_plane(points):
    """Fit a plane to points using RANSAC."""
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.005,
            ransac_n=3,
            num_iterations=2000
        )
        
        if len(inliers) < 100:
            print("Insufficient inliers for plane fitting, falling back to bounding box")
            obb = pcd.get_oriented_bounding_box()
            return None, None, obb
        
        return plane_model, inliers, None
    except Exception as e:
        raise ValueError(f"Error in fit_plane: {str(e)}")

def estimate_pose(points, inliers, plane_model, obb=None):
    """Estimate box pose from plane fit or bounding box."""
    try:
        if plane_model is not None and inliers is not None:
            plane_points = points[inliers]
            
            normal = plane_model[:3]
            normal = normal / np.linalg.norm(normal)
            center = np.mean(plane_points, axis=0)
            
            # Use PCA to determine x and y axes
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(plane_points)
            mean, cov = pcd.compute_mean_and_covariance()
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            x_axis = eigenvectors[:, 2]  # Largest principal direction
            
            z_axis = normal
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            T = np.eye(4)
            T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
            T[:3, 3] = center
        elif obb is not None:
            rotation = obb.R
            center = np.asarray(obb.center)
            T = np.eye(4)
            T[:3, :3] = rotation
            T[:3, 3] = center
        else:
            raise ValueError("Neither plane model nor bounding box provided")
        
        return T
    except Exception as e:
        raise ValueError(f"Error in estimate_pose: {str(e)}")

def visualize_result(points, colors, T, extrinsics=None, use_world_frame=True):
    """Visualize point cloud and estimated box pose."""
    try:
        if len(points) == 0:
            raise ValueError("Point cloud is empty for visualization")
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            raise ValueError("Point cloud contains NaN or Inf values")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd_bbox = pcd.get_axis_aligned_bounding_box()
        pcd_center = pcd_bbox.get_center()
        box_size = pcd_bbox.get_extent()
        print(f"Point cloud center: {pcd_center}")
        print(f"Computed box size from point cloud: {box_size}")
        
        box = o3d.geometry.TriangleMesh.create_box(
            width=box_size[0],
            height=box_size[1],
            depth=box_size[2]
        )
        box.compute_vertex_normals()
        box.paint_uniform_color([1, 0, 0])
        
        box.translate(-np.array(box_size) / 2.0)
        
        # Apply transformation
        box.transform(T)
        
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        frame.transform(T)
        
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        if use_world_frame and extrinsics is not None:
            camera_frame.transform(extrinsics)
        
        box_bbox = box.get_axis_aligned_bounding_box()
        box_center = box_bbox.get_center()
        print(f"Box center after transformation: {box_center}")
        
        combined_geometries = [pcd, box, frame, camera_frame]
        combined_points = np.vstack([np.asarray(geo.get_axis_aligned_bounding_box().get_box_points()) for geo in combined_geometries])
        combined_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(combined_points))
        combined_center = combined_bbox.get_center()
        combined_extent = combined_bbox.get_extent()
        print(f"Combined bounding box center: {combined_center}")
        print(f"Combined bounding box extent: {combined_extent}")
        
        # Visualize point cloud only
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Debug")
        vis.add_geometry(pcd)
        
        ctr = vis.get_view_control()
        ctr.set_lookat(pcd_center)
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.3)
        
        print("Visualizing point cloud only...")
        vis.run()
        vis.destroy_window()
        
        # Visualize point cloud and box only
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud and Box")
        vis.add_geometry(pcd)
        vis.add_geometry(box)
        
        ctr = vis.get_view_control()
        ctr.set_lookat(combined_center)
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        max_extent = np.max(combined_extent)
        zoom = 2.0 / max_extent
        ctr.set_zoom(zoom)
        
        print("Visualizing point cloud and box...")
        vis.run()
        vis.destroy_window()
        
        # Visualize all geometries
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Full Scene")
        vis.add_geometry(pcd)
        vis.add_geometry(box)
        vis.add_geometry(frame)
        vis.add_geometry(camera_frame)
        
        ctr = vis.get_view_control()
        ctr.set_lookat(combined_center)
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(zoom)
        
        print("Visualizing all geometries...")
        vis.run()
        vis.destroy_window()
    except Exception as e:
        raise ValueError(f"Error in visualize_result: {str(e)}")

def main(extrinsic, intrinsics, color_img, depth_map):
    """Main function to estimate box pose with robust outlier rejection."""
    try:
        if not all([extrinsic.shape == (4,4), intrinsics.shape == (3,3)]):
            raise ValueError("Invalid extrinsic or intrinsic matrix shapes")
        if not np.allclose(extrinsic[3, :], [0, 0, 0, 1]):
            raise ValueError("Invalid extrinsics matrix: last row must be [0, 0, 0, 1]")
            
        points, colors = segment_box(depth_map, color_img, intrinsics)
        
        if len(points) < 100:
            raise ValueError(f"Insufficient valid points after segmentation: {len(points)}")
            
        print(f"Point cloud stats: min={points.min(axis=0)}, max={points.max(axis=0)}")
            
        plane_model, inliers, obb = fit_plane(points)
        
        T_camera = estimate_pose(points, inliers, plane_model, obb)
        
        T_world = extrinsic @ T_camera
        
        print(f"Camera to object transform:\n{T_camera}")
        print(f"Extrinsic matrix:\n{extrinsic}")
        
        print("Visualizing in camera frame...")
        visualize_result(points, colors, T_camera, extrinsics=extrinsic, use_world_frame=False)
        
        return T_world
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        extrinsic = np.load('data/extrinsics.npy')
        intrinsics = np.load('data/intrinsics.npy')
        one_box_color = np.load('data/one-box.color.npdata.npy')
        one_box_depth = np.load('data/one-box.depth.npdata.npy')
        
        transformation_matrix = main(extrinsic, intrinsics, one_box_color, one_box_depth)
        
        if transformation_matrix is not None:
            print("Estimated Transformation Matrix (World to Object):")
            print(transformation_matrix)
        else:
            print("Pose estimation failed")
            
    except FileNotFoundError as e:
        print(f"Error loading data files: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")