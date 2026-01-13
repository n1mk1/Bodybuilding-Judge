import cv2
import numpy as np
import mediapipe as mp
import os
import shutil


class PoseClassifier:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
    def classify_pose(self, landmarks, w, h):
        def px(lm):
            return [int(lm.x * w), int(lm.y * h)]
        
        ls = px(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        rs = px(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER])
        le = px(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW])
        re = px(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW])
        lw = px(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST])
        rw = px(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        
        left_angle = self._angle(ls, le, lw)
        right_angle = self._angle(rs, re, rw)
        
        arms_up = lw[1] < le[1] and rw[1] < re[1]
        flexing = 40 < left_angle < 110 and 40 < right_angle < 110
        
        if flexing and arms_up:
            return "front_double_bicep" if self._facing_front(landmarks) else "back_double_bicep"
        
        return "unknown_pose"
    
    def _angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(rad * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def _facing_front(self, landmarks):
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        l_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        r_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        avg_vis = (nose.visibility + l_ear.visibility + r_ear.visibility) / 3
        if avg_vis < 0.5:
            return False
        
        return not (nose.z > l_ear.z and nose.z > r_ear.z)


class ImageOrganizer:
    def __init__(self, working_dir):
        self.working_directory = working_dir
        self.classifier = PoseClassifier()
        self.mp_pose = mp.solutions.pose
        self.formats = ('.jpg', '.jpeg', '.png')
        
    def get_unorganized_images(self):
        if not os.path.exists(self.working_directory):
            return []
        
        return [f for f in os.listdir(self.working_directory) 
                if f.lower().endswith(self.formats) and not f.startswith('athlete')]
    
    def classify_images(self, images):
        classified = []
        unclassified = []
        
        with self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
            for img_file in images:
                img_path = os.path.join(self.working_directory, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    unclassified.append(img_file)
                    continue
                
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                
                if results.pose_landmarks:
                    pose_type = self.classifier.classify_pose(results.pose_landmarks.landmark, w, h)
                    if pose_type != "unknown_pose":
                        classified.append({'original': img_file, 'pose': pose_type, 'path': img_path})
                    else:
                        unclassified.append(img_file)
                else:
                    unclassified.append(img_file)
        
        return classified, unclassified
    
    def group_by_pose(self, classified):
        poses = {}
        for item in classified:
            p = item['pose']
            if p not in poses:
                poses[p] = []
            poses[p].append(item)
        return poses
    
    def assign_to_athletes(self, poses_dict):
        a1_poses = {}
        a2_poses = {}
        
        for pose_type, items in poses_dict.items():
            if len(items) >= 1:
                a1_poses[pose_type] = items[0]
            if len(items) >= 2:
                a2_poses[pose_type] = items[1]
        
        return a1_poses, a2_poses
    
    def check_existing_files(self, a1_poses, a2_poses):
        files = []
        for pt in a1_poses.keys():
            files.append(f"athlete1_{pt}.jpg")
        for pt in a2_poses.keys():
            files.append(f"athlete2_{pt}.jpg")
        
        return [f for f in files if os.path.exists(os.path.join(self.working_directory, f))]
    
    def organize_files(self, a1_poses, a2_poses):
        count = 0
        errors = []
        
        for pose_type, item in a1_poses.items():
            new_path = os.path.join(self.working_directory, f"athlete1_{pose_type}.jpg")
            try:
                shutil.copy2(item['path'], new_path)
                count += 1
            except Exception as e:
                errors.append(f"{item['original']}: {str(e)}")
        
        for pose_type, item in a2_poses.items():
            new_path = os.path.join(self.working_directory, f"athlete2_{pose_type}.jpg")
            try:
                shutil.copy2(item['path'], new_path)
                count += 1
            except Exception as e:
                errors.append(f"{item['original']}: {str(e)}")
        
        return count, errors
    
    def organize_poses(self, overwrite=False):
        images = self.get_unorganized_images()
        if not images:
            return {'success': False, 'message': 'No unorganized images found in directory.', 
                    'classified_count': 0, 'unclassified_count': 0}
        
        classified, unclassified = self.classify_images(images)
        
        if not classified:
            return {'success': False, 'message': f'No poses could be identified from {len(images)} images.',
                    'classified_count': 0, 'unclassified_count': len(unclassified), 'unclassified': unclassified}
        
        poses = self.group_by_pose(classified)
        a1_poses, a2_poses = self.assign_to_athletes(poses)
        
        existing = self.check_existing_files(a1_poses, a2_poses)
        if existing and not overwrite:
            return {'success': False, 'message': 'Organized files already exist. Set overwrite=True to replace.',
                    'existing_files': existing, 'classified_count': len(classified), 'unclassified_count': len(unclassified)}
        
        count, errors = self.organize_files(a1_poses, a2_poses)
        
        return {'success': True, 'renamed_count': count, 'athlete1_count': len(a1_poses),
                'athlete2_count': len(a2_poses), 'poses_found': list(poses.keys()),
                'unclassified_count': len(unclassified), 'unclassified': unclassified, 'errors': errors}


class AthleteLoader:
    def __init__(self, working_dir):
        self.working_directory = working_dir
        self.formats = ('.jpg', '.jpeg', '.png')
        
    def load_athletes(self):
        athletes = {}
        
        if not os.path.exists(self.working_directory):
            return athletes
        
        for file in os.listdir(self.working_directory):
            if file.lower().endswith(self.formats):
                parts = file.split('_')
                if len(parts) >= 2:
                    athlete = parts[0]
                    pose = '_'.join(parts[1:]).rsplit('.', 1)[0]
                    
                    if athlete not in athletes:
                        athletes[athlete] = {}
                    athletes[athlete][pose] = os.path.join(self.working_directory, file)
        
        return athletes
    
    def get_athlete_names(self, athletes_dict):
        return list(athletes_dict.keys())
    
    def get_image_path(self, athletes_dict, athlete_name, pose_name):
        if athlete_name in athletes_dict:
            if pose_name in athletes_dict[athlete_name]:
                return athletes_dict[athlete_name][pose_name]
        return None
