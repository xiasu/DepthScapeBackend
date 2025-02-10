from DepthScape_Classes.Geometry.Plane import Plane
import numpy as np
class Face:
    def __init__(self, depthScape, results):
        self.depthScape = depthScape
        self.results = results
        if not self.results.pose_landmarks:
            raise ValueError("No face landmarks detected")
        self.calculate_3D_positions()
    def calculate_3D_positions(self):
        # Get the point cloud from depthScape
        points = self.depthScape.points
        # Initialize dictionary to store 3D positions
        positions_3d = {}
        visible_depths = []
        image_height, image_width = self.depthScape.image.shape[:2]

        # First pass - get depths for visible landmarks
        for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
            # Only process facial points (0-10)
            if idx <= 10 and landmark.visibility >= 0.995:
                # Convert normalized coordinates to pixel coordinates
                pixel_x = int(landmark.x * image_width)
                pixel_y = int(landmark.y * image_height)

                # Ensure coordinates are within image bounds
                if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
                    # Get corresponding 3D point from point cloud
                    point_3d = points[pixel_y][pixel_x]
                    if point_3d is not None:
                        positions_3d[idx] = point_3d
                        visible_depths.append(point_3d[2])  # Store Z coordinate

        # Calculate linear transformation parameters for depth mapping if we have visible points
        if visible_depths:
            # Get corresponding landmark.z values for visible points
            visible_landmark_z = []
            visible_point_z = []
            for idx, point_3d in positions_3d.items():
                landmark = self.results.pose_landmarks.landmark[idx]
                visible_landmark_z.append(landmark.z)
                visible_point_z.append(point_3d[2])

            # Calculate linear regression parameters (multiplier and shift)
            visible_landmark_z = np.array(visible_landmark_z)
            visible_point_z = np.array(visible_point_z)
            A = np.vstack([visible_landmark_z, np.ones(len(visible_landmark_z))]).T
            multiplier, shift = np.linalg.lstsq(A, visible_point_z, rcond=None)[0]
        else:
            return None

        # Second pass - estimate positions for non-visible facial landmarks using the transformation
        for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
            if idx <= 10 and idx not in positions_3d:
                # Convert normalized coordinates to pixel coordinates
                pixel_x = int(landmark.x * image_width)
                pixel_y = int(landmark.y * image_height)
                
                # Use linear transformation to estimate depth
                estimated_depth = multiplier * landmark.z + shift
                
                # Create estimated 3D position
                if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
                    # Get X,Y from point cloud
                    point_3d = points[pixel_y][pixel_x]
                    if point_3d is not None:
                        positions_3d[idx] = np.array([point_3d[0], point_3d[1], estimated_depth])
        self.positions_3d = positions_3d
        return positions_3d

    def get_cranial(self):
        # Get key positions
        nose_pos = self.positions_3d.get(0)  # Nose
        left_eye_pos = self.positions_3d.get(2)  # Left eye
        right_eye_pos = self.positions_3d.get(5)  # Right eye
        left_mouth_pos = self.positions_3d.get(9)  # Left mouth
        right_mouth_pos = self.positions_3d.get(10)  # Right mouth

        # Calculate midpoints
        eye_center = np.array([
            (left_eye_pos[0] + right_eye_pos[0]) / 2,
            (left_eye_pos[1] + right_eye_pos[1]) / 2,
            (left_eye_pos[2] + right_eye_pos[2]) / 2
        ])
        
        mouth_center = np.array([
            (left_mouth_pos[0] + right_mouth_pos[0]) / 2,
            (left_mouth_pos[1] + right_mouth_pos[1]) / 2,
            (left_mouth_pos[2] + right_mouth_pos[2]) / 2
        ])

        # Calculate cranial vector (from mouth center to eye center)
        cranial = eye_center - mouth_center
        cranial = cranial / np.linalg.norm(cranial)
        return cranial


    def get_anterior(self):
        # Get key positions
        left_eye_pos = self.positions_3d.get(2)  # Left eye
        right_eye_pos = self.positions_3d.get(5)  # Right eye
        left_mouth_pos = self.positions_3d.get(9)  # Left mouth
        right_mouth_pos = self.positions_3d.get(10)  # Right mouth

        if not all([left_eye_pos is not None, right_eye_pos is not None, 
                   left_mouth_pos is not None, right_mouth_pos is not None]):
            return None

        # Calculate vectors forming the quadrilateral
        v1 = np.array(right_eye_pos) - np.array(left_eye_pos)
        v2 = np.array(right_mouth_pos) - np.array(left_mouth_pos)
        v3 = np.array(left_mouth_pos) - np.array(left_eye_pos)
        v4 = np.array(right_mouth_pos) - np.array(right_eye_pos)

        # Calculate anterior vector as normal to the quadrilateral
        anterior = np.cross(v1, v3) + np.cross(v2, v4)
        anterior = anterior / np.linalg.norm(anterior)
        return anterior


    def get_median(self):
        # Get key positions
        nose_pos = self.positions_3d.get(0)  # Nose
        left_eye_pos = self.positions_3d.get(2)  # Left eye
        right_eye_pos = self.positions_3d.get(5)  # Right eye
        left_mouth_pos = self.positions_3d.get(9)  # Left mouth
        right_mouth_pos = self.positions_3d.get(10)  # Right mouth

        # Get anterior and cranial vectors
        anterior = self.get_anterior()
        cranial = self.get_cranial()

        # Calculate midpoints
        eye_center = np.array([
            (left_eye_pos[0] + right_eye_pos[0]) / 2,
            (left_eye_pos[1] + right_eye_pos[1]) / 2,
            (left_eye_pos[2] + right_eye_pos[2]) / 2
        ])
        
        mouth_center = np.array([
            (left_mouth_pos[0] + right_mouth_pos[0]) / 2,
            (left_mouth_pos[1] + right_mouth_pos[1]) / 2,
            (left_mouth_pos[2] + right_mouth_pos[2]) / 2
        ])

        # Calculate center and spans
        center = (eye_center + mouth_center) / 2
        vertical_span = np.linalg.norm(eye_center - mouth_center)*2

        # Calculate median vector (cross product of anterior and cranial)
        median = np.cross(anterior, cranial)
        median = median / np.linalg.norm(median)

        # Create plane parameters
        plane = Plane(median[0], median[1], median[2],
                     -(median[0] * center[0] + median[1] * center[1] + median[2] * center[2]),
                     np.array(list(self.positions_3d.values())))
        plane.set_primary_center_span(np.array([anterior, cranial]), center, np.array([vertical_span, vertical_span]))
        return plane

    def get_frontal(self):
        cranial = self.get_cranial()
        anterior = self.get_anterior()


        # Get key positions
        left_eye_pos = self.positions_3d.get(2)  # Left eye
        right_eye_pos = self.positions_3d.get(5)  # Right eye
        left_mouth_pos = self.positions_3d.get(9)  # Left mouth
        right_mouth_pos = self.positions_3d.get(10)  # Right mouth

        # Calculate midpoints and center
        eye_center = np.array([
            (left_eye_pos[0] + right_eye_pos[0]) / 2,
            (left_eye_pos[1] + right_eye_pos[1]) / 2,
            (left_eye_pos[2] + right_eye_pos[2]) / 2
        ])
        
        mouth_center = np.array([
            (left_mouth_pos[0] + right_mouth_pos[0]) / 2,
            (left_mouth_pos[1] + right_mouth_pos[1]) / 2,
            (left_mouth_pos[2] + right_mouth_pos[2]) / 2
        ])

        center = (eye_center + mouth_center) / 2

        # Calculate spans
        eye_span = np.linalg.norm(np.array(right_eye_pos) - np.array(left_eye_pos))*2
        vertical_span = np.linalg.norm(eye_center - mouth_center)*2

        # Calculate lateral vector (cross product of anterior and cranial)
        lateral = np.cross(anterior, cranial)
        lateral = lateral / np.linalg.norm(lateral)

        # Create plane parameters using anterior as normal, lateral as x and cranial as y
        plane = Plane(anterior[0], anterior[1], anterior[2],
                     -(anterior[0] * center[0] + anterior[1] * center[1] + anterior[2] * center[2]),
                     np.array(list(self.positions_3d.values())))
        plane.set_primary_center_span(np.array([lateral, cranial]), center, np.array([eye_span, vertical_span]))
        return plane