from DepthScape_Classes.Geometry.Plane import Plane
import numpy as np
class Skeleton:
    def __init__(self, depthScape, results):
        self.depthScape = depthScape
        #This results is the output of the openpose model
        self.results = results
        self.calculate_3D_positions()
        
    def calculate_3D_positions(self):
        #This method should calculate the 3D positions of the landmarks
        #It should return a dictionary with the landmark name as the key and the 3D position as the value
        # Get the point cloud from depthScape
        points = self.depthScape.points
        if not self.results.pose_landmarks:
            return None


        # Initialize dictionary to store 3D positions
        positions_3d = {}
        visible_depths = []
        image_height, image_width = self.depthScape.image.shape[:2]

        # First pass - get depths for visible landmarks
        for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
            # Face points (0-10) use higher threshold, body points use lower threshold
            if (idx <= 10 and landmark.visibility >= 0.995) or (idx > 10 and landmark.visibility >= 0.9):
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

        # Second pass - estimate positions for non-visible landmarks using the transformation
        for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
            if idx not in positions_3d:
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

        if not self.positions_3d:
            return None
            
        # Get the shoulder and hip positions
        left_shoulder_pos = self.positions_3d.get(11)  # Left shoulder
        right_shoulder_pos = self.positions_3d.get(12)  # Right shoulder
        left_hip_pos = self.positions_3d.get(23)  # Left hip
        right_hip_pos = self.positions_3d.get(24)  # Right hip
        left_ear_pos = self.positions_3d.get(7)  # Left ear
        right_ear_pos = self.positions_3d.get(8)  # Right ear

        # Calculate shoulder midpoint
        shoulder_midpoint = [(left_shoulder_pos[0] + right_shoulder_pos[0])/2,
                           (left_shoulder_pos[1] + right_shoulder_pos[1])/2,
                           (left_shoulder_pos[2] + right_shoulder_pos[2])/2]

        # Check visibility of hip points
        left_hip_visible = self.results.pose_landmarks.landmark[23].visibility > 0.9
        right_hip_visible = self.results.pose_landmarks.landmark[24].visibility > 0.9

        if left_hip_visible and right_hip_visible:
            # Use hip to shoulder vector
            hip_midpoint = [(left_hip_pos[0] + right_hip_pos[0])/2,
                           (left_hip_pos[1] + right_hip_pos[1])/2,
                           (left_hip_pos[2] + right_hip_pos[2])/2]
            
            cranial_vector = [shoulder_midpoint[0] - hip_midpoint[0],
                            shoulder_midpoint[1] - hip_midpoint[1],
                            shoulder_midpoint[2] - hip_midpoint[2]]
        else:
            # Use shoulder to ear vector instead
            ear_midpoint = [(left_ear_pos[0] + right_ear_pos[0])/2,
                           (left_ear_pos[1] + right_ear_pos[1])/2,
                           (left_ear_pos[2] + right_ear_pos[2])/2]
            
            cranial_vector = [ear_midpoint[0] - shoulder_midpoint[0],
                            ear_midpoint[1] - shoulder_midpoint[1],
                            ear_midpoint[2] - shoulder_midpoint[2]]

        # Normalize the vector
        magnitude = (cranial_vector[0]**2 + cranial_vector[1]**2 + cranial_vector[2]**2)**0.5
        if magnitude > 0:
            cranial_vector = [x/magnitude for x in cranial_vector]
            return cranial_vector
        return None

    def get_anterior(self):
        if not self.positions_3d:
            return None
            
        # Get all required points
        left_shoulder_pos = self.positions_3d.get(11)  # Left shoulder
        right_shoulder_pos = self.positions_3d.get(12)  # Right shoulder
        left_hip_pos = self.positions_3d.get(23)  # Left hip
        right_hip_pos = self.positions_3d.get(24)  # Right hip
        left_ear_pos = self.positions_3d.get(7)  # Left ear
        right_ear_pos = self.positions_3d.get(8)  # Right ear

        # Check visibility of hip points
        left_hip_visible = self.results.pose_landmarks.landmark[23].visibility > 0.9
        right_hip_visible = self.results.pose_landmarks.landmark[24].visibility > 0.9

        if left_hip_visible and right_hip_visible:
            # Calculate vectors forming the quadrilateral
            v1 = [left_shoulder_pos[i] - right_shoulder_pos[i] for i in range(3)]  # Right to left shoulder
            v2 = [left_hip_pos[i] - left_shoulder_pos[i] for i in range(3)]  # Left shoulder to left hip
            v3 = [right_hip_pos[i] - right_shoulder_pos[i] for i in range(3)]  # Right shoulder to right hip
            
            # Calculate normal vector using cross product of diagonals
            d1 = [left_hip_pos[i] - right_shoulder_pos[i] for i in range(3)]  # Diagonal 1
            d2 = [right_hip_pos[i] - left_shoulder_pos[i] for i in range(3)]  # Diagonal 2
            
            anterior_vector = [
                d1[1] * d2[2] - d1[2] * d2[1],
                d1[2] * d2[0] - d1[0] * d2[2],
                d1[0] * d2[1] - d1[1] * d2[0]
            ]
        else:
            # Use shoulders and ears when hips not visible
            v1 = [left_shoulder_pos[i] - right_shoulder_pos[i] for i in range(3)]  # Right to left shoulder
            v2 = [left_ear_pos[i] - left_shoulder_pos[i] for i in range(3)]  # Left shoulder to left ear
            v3 = [right_ear_pos[i] - right_shoulder_pos[i] for i in range(3)]  # Right shoulder to right ear
            
            # Calculate normal vector using cross product of diagonals
            d1 = [left_ear_pos[i] - right_shoulder_pos[i] for i in range(3)]  # Diagonal 1
            d2 = [right_ear_pos[i] - left_shoulder_pos[i] for i in range(3)]  # Diagonal 2
            
            anterior_vector = [
                d1[1] * d2[2] - d1[2] * d2[1],
                d1[2] * d2[0] - d1[0] * d2[2],
                d1[0] * d2[1] - d1[1] * d2[0]
            ]

        # Normalize the vector
        magnitude = (anterior_vector[0]**2 + anterior_vector[1]**2 + anterior_vector[2]**2)**0.5
        if magnitude > 0:
            normalized = [x/magnitude for x in anterior_vector]
            # Check if vector points towards positive z and flip if not
            if normalized[2] > 0:
                normalized = [-x for x in normalized]
            return normalized
        return None

    def get_median(self):
        # Get cranial and anterior vectors to define plane orientation
        cranial = self.get_cranial()
        anterior = self.get_anterior()
        
        if not (cranial and anterior):
            return None
            
        # Calculate lateral (medial-lateral) vector as cross product
        lateral = [
            cranial[1] * anterior[2] - cranial[2] * anterior[1],
            cranial[2] * anterior[0] - cranial[0] * anterior[2], 
            cranial[0] * anterior[1] - cranial[1] * anterior[0]
        ]
        
        # Get midpoints between shoulders and hips
        left_shoulder_pos = self.positions_3d.get(11)  # Left shoulder
        right_shoulder_pos = self.positions_3d.get(12)  # Right shoulder
        left_hip_pos = self.positions_3d.get(23)  # Left hip
        right_hip_pos = self.positions_3d.get(24)  # Right hip
        left_ear_pos = self.positions_3d.get(7)  # Left ear
        right_ear_pos = self.positions_3d.get(8)  # Right ear

        # Check visibility of points
        left_shoulder_visible = self.results.pose_landmarks.landmark[11].visibility > 0.9
        right_shoulder_visible = self.results.pose_landmarks.landmark[12].visibility > 0.9
        left_hip_visible = self.results.pose_landmarks.landmark[23].visibility > 0.9
        right_hip_visible = self.results.pose_landmarks.landmark[24].visibility > 0.9

        if left_shoulder_visible and right_shoulder_visible:
            shoulder_center = [
                (left_shoulder_pos[0] + right_shoulder_pos[0]) / 2,
                (left_shoulder_pos[1] + right_shoulder_pos[1]) / 2,
                (left_shoulder_pos[2] + right_shoulder_pos[2]) / 2
            ]

            # Calculate center point based on visibility of hips
            if left_hip_visible and right_hip_visible:
                hip_center = [
                    (left_hip_pos[0] + right_hip_pos[0]) / 2,
                    (left_hip_pos[1] + right_hip_pos[1]) / 2,
                    (left_hip_pos[2] + right_hip_pos[2]) / 2
                ]
                center = [
                    (shoulder_center[0] + hip_center[0]) / 2,
                    (shoulder_center[1] + hip_center[1]) / 2,
                    (shoulder_center[2] + hip_center[2]) / 2
                ]
            else:
                center = shoulder_center

            # Calculate ear midpoint to determine plane dimensions
            
            ear_center = [
                (left_ear_pos[0] + right_ear_pos[0]) / 2,
                (left_ear_pos[1] + right_ear_pos[1]) / 2,
                (left_ear_pos[2] + right_ear_pos[2]) / 2
            ]
            
            # Calculate distance from center to ear midpoint
            span = ((ear_center[0] - center[0])**2 + 
                    (ear_center[1] - center[1])**2 + 
                    (ear_center[2] - center[2])**2)**0.5

            # Calculate plane parameters using anterior as x and cranial as y
            plane = Plane(lateral[0], lateral[1], lateral[2], 
                        -(lateral[0] * center[0] + lateral[1] * center[1] + lateral[2] * center[2]),
                        np.array(list(self.positions_3d.values())))
            plane.set_primary_center_span(np.array([anterior,cranial]),np.array(center),np.array([span,span]))
            return plane

        return None

    def get_frontal(self):
        cranial = self.get_cranial()
        anterior = self.get_anterior()
        
        if not (cranial and anterior):
            return None
        # Get key positions
        left_shoulder_pos = self.positions_3d.get(11)  # Left shoulder
        right_shoulder_pos = self.positions_3d.get(12)  # Right shoulder
        left_hip_pos = self.positions_3d.get(23)  # Left hip
        right_hip_pos = self.positions_3d.get(24)  # Right hip
        left_ear_pos = self.positions_3d.get(7)  # Left ear
        right_ear_pos = self.positions_3d.get(8)  # Right ear

        # Check visibility of points
        left_shoulder_visible = self.results.pose_landmarks.landmark[11].visibility > 0.9
        right_shoulder_visible = self.results.pose_landmarks.landmark[12].visibility > 0.9
        left_hip_visible = self.results.pose_landmarks.landmark[23].visibility > 0.9
        right_hip_visible = self.results.pose_landmarks.landmark[24].visibility > 0.9

        shoulder_center = [
            (left_shoulder_pos[0] + right_shoulder_pos[0]) / 2,
            (left_shoulder_pos[1] + right_shoulder_pos[1]) / 2,
            (left_shoulder_pos[2] + right_shoulder_pos[2]) / 2
        ]

        # Calculate center point based on visibility of hips
        if left_hip_visible and right_hip_visible:
            hip_center = [
                (left_hip_pos[0] + right_hip_pos[0]) / 2,
                (left_hip_pos[1] + right_hip_pos[1]) / 2,
                (left_hip_pos[2] + right_hip_pos[2]) / 2
            ]
            center = [
                (shoulder_center[0] + hip_center[0]) / 2,
                (shoulder_center[1] + hip_center[1]) / 2,
                (shoulder_center[2] + hip_center[2]) / 2
            ]
        else:
            center = shoulder_center

        # Calculate shoulder span for x scale
        shoulder_span = ((left_shoulder_pos[0] - right_shoulder_pos[0])**2 + 
                        (left_shoulder_pos[1] - right_shoulder_pos[1])**2 + 
                        (left_shoulder_pos[2] - right_shoulder_pos[2])**2)**0.5

        # Calculate ear midpoint for y scale
        ear_center = [
            (left_ear_pos[0] + right_ear_pos[0]) / 2,
            (left_ear_pos[1] + right_ear_pos[1]) / 2,
            (left_ear_pos[2] + right_ear_pos[2]) / 2
        ]
        
        # Calculate distance from center to ear midpoint for y scale
        vertical_span = ((ear_center[0] - center[0])**2 + 
                        (ear_center[1] - center[1])**2 + 
                        (ear_center[2] - center[2])**2)**0.5

        # Calculate frontal plane normal (cross product of cranial and anterior)
        frontal = np.cross(cranial, anterior)
        
        # Calculate plane parameters using frontal as x and cranial as y
        plane = Plane(frontal[0], frontal[1], frontal[2],
                    -(frontal[0] * center[0] + frontal[1] * center[1] + frontal[2] * center[2]),
                    np.array(list(self.positions_3d.values())))
        plane.set_primary_center_span(np.array([frontal,cranial]),np.array(center),np.array([shoulder_span,vertical_span]))
        return plane