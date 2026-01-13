import cv2
import numpy as np
import mediapipe as mp

COLOR_SHOULDER = (255, 100, 0)
COLOR_LAT = (0, 165, 255)
COLOR_WAIST = (0, 0, 255)
COLOR_QUAD = (255, 255, 0)
DOT_COLOR = (0, 255, 0)
LINE_THICKNESS = 3
DOT_RADIUS = 6

def get_mask_and_landmarks(image):
    mp_pose = mp.solutions.pose
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        landmarks = results.pose_landmarks

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_seg:
        seg_results = selfie_seg.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask = (seg_results.segmentation_mask > 0.5).astype(np.uint8) * 255

    return mask, landmarks

def separate_arms_from_torso(mask, landmarks, w, h):
    l_sh = (int(landmarks.landmark[11].x * w), int(landmarks.landmark[11].y * h))
    r_sh = (int(landmarks.landmark[12].x * w), int(landmarks.landmark[12].y * h))
    l_elbow = (int(landmarks.landmark[13].x * w), int(landmarks.landmark[13].y * h))
    r_elbow = (int(landmarks.landmark[14].x * w), int(landmarks.landmark[14].y * h))

    cut_thickness = int(w * 0.03)
    
    cv2.line(mask, l_sh, l_elbow, 0, cut_thickness)
    cv2.line(mask, r_sh, r_elbow, 0, cut_thickness)
    return mask

def find_edges_center_out(mask, y, center_x):
    h, w = mask.shape
    if y >= h or y < 0: 
        return None, None
    center_x = max(0, min(center_x, w - 1))
    
    if mask[y, center_x] == 0: 
        return None, None 

    row = mask[y, :]
    
    left_seg = row[:center_x][::-1]
    zeros_l = np.where(left_seg == 0)[0]
    x1 = (center_x - zeros_l[0]) if len(zeros_l) > 0 else 0
    
    right_seg = row[center_x:]
    zeros_r = np.where(right_seg == 0)[0]
    x2 = (center_x + zeros_r[0]) if len(zeros_r) > 0 else w - 1
    
    return x1, x2

def find_outer_sweep_edges(mask, y):
    h, w = mask.shape
    if y >= h or y < 0: 
        return None, None

    row = mask[y, :]
    indices = np.where(row > 0)[0]

    if len(indices) == 0: 
        return None, None

    return indices[0], indices[-1]

def draw_measurement(img, y, x1, x2, color, label=None):
    cv2.line(img, (x1, y), (x2, y), color, LINE_THICKNESS)
    cv2.circle(img, (x1, y), DOT_RADIUS, DOT_COLOR, -1)
    cv2.circle(img, (x2, y), DOT_RADIUS, DOT_COLOR, -1)
    
    width = abs(x2 - x1)
    if label:
        cv2.putText(img, f"{label}", (x2 + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return width

def analyze_xframe(image_path): # god HELP MEEE PLS, Hour of work count = 47
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return None

    h, w, _ = img.shape
    mask, landmarks = get_mask_and_landmarks(img)
    if not landmarks:
        print("Error: No pose landmarks detected")
        return None

    l_sh = landmarks.landmark[11]
    r_sh = landmarks.landmark[12]
    l_hip = landmarks.landmark[23]
    r_hip = landmarks.landmark[24]
    
    shoulder_y = int((l_sh.y + r_sh.y) / 2 * h)
    hip_y = int((l_hip.y + r_hip.y) / 2 * h)
    knee_y = int((landmarks.landmark[25].y + landmarks.landmark[26].y) / 2 * h)
    
    spine_x = int((l_sh.x + r_sh.x + l_hip.x + r_hip.x) / 4 * w)
    torso_length = hip_y - shoulder_y

    mask = separate_arms_from_torso(mask, landmarks, w, h)

    delt_y = shoulder_y + int(torso_length * 0.05)
    s_x1, s_x2 = find_outer_sweep_edges(mask, delt_y)
    if s_x1 is None: 
        s_x1, s_x2 = (spine_x, spine_x)

    lat_start = shoulder_y + int(torso_length * 0.25)
    lat_end = shoulder_y + int(torso_length * 0.45)
    
    max_lat_width = 0
    best_lat_y = lat_start
    best_lat_coords = (spine_x, spine_x)

    for y in range(lat_start, lat_end, 5):
        x1, x2 = find_edges_center_out(mask, y, spine_x)
        if x1 is not None:
            width = x2 - x1
            if width > max_lat_width:
                max_lat_width = width
                best_lat_y = y
                best_lat_coords = (x1, x2)

    waist_start = lat_end
    waist_end = hip_y - int(torso_length * 0.05)
    
    min_waist_width = float('inf')
    best_waist_y = waist_start
    best_waist_coords = (spine_x, spine_x)

    for y in range(waist_start, waist_end, 5):
        x1, x2 = find_edges_center_out(mask, y, spine_x)
        if x1 is not None:
            width = x2 - x1
            if width < min_waist_width and width > (max_lat_width * 0.3):
                min_waist_width = width
                best_waist_y = y
                best_waist_coords = (x1, x2)

    quad_start = hip_y
    quad_end = knee_y - int((knee_y - hip_y) * 0.20)
    
    max_quad_width = 0
    best_quad_y = quad_start
    best_quad_coords = (spine_x, spine_x)

    for y in range(quad_start, quad_end, 5):
        qx1, qx2 = find_outer_sweep_edges(mask, y)
        if qx1 is not None:
            width = qx2 - qx1
            if width > max_quad_width:
                max_quad_width = width
                best_quad_y = y
                best_quad_coords = (qx1, qx2)

    sh_px = draw_measurement(img, delt_y, s_x1, s_x2, COLOR_SHOULDER, "Shldr")
    lat_px = draw_measurement(img, best_lat_y, *best_lat_coords, COLOR_LAT, "Lat")
    waist_px = draw_measurement(img, best_waist_y, *best_waist_coords, COLOR_WAIST, "Waist")
    quad_px = draw_measurement(img, best_quad_y, *best_quad_coords, COLOR_QUAD, "Quad")

    if waist_px > 0:
        x_frame_score = (lat_px + quad_px) / waist_px
        mass_score = (sh_px + lat_px + quad_px) / waist_px
    else:
        x_frame_score = 0
        mass_score = 0

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (400, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Shoulders: {sh_px} px", (20, 30), font, 0.7, COLOR_SHOULDER, 2)
    cv2.putText(img, f"Lats: {lat_px} px", (20, 60), font, 0.7, COLOR_LAT, 2)
    cv2.putText(img, f"Waist: {waist_px} px", (20, 90), font, 0.7, COLOR_WAIST, 2)
    cv2.putText(img, f"Quads: {quad_px} px", (20, 120), font, 0.7, COLOR_QUAD, 2)
    
    cv2.line(img, (20, 140), (380, 140), (255,255,255), 1)
    
    cv2.putText(img, f"X-Frame Ratio (Lat + Quad) / Waist: {x_frame_score:.2f}", (20, 170), font, 0.5, (255, 255, 255), 2)
    cv2.putText(img, f"Mass Score (Shdr + Lat + Quad) / Waist: {mass_score:.2f}", (20, 230), font, 0.5, (255, 255, 255), 2)
    
    return {
        'metrics': {
            'x_frame': x_frame_score,
            'mass': mass_score,
            'shoulder': sh_px,
            'lat': lat_px,
            'waist': waist_px,
            'quad': quad_px
        },
        'annotated_image': img
    }


if __name__ == "__main__":
    IMAGE_NAME = "athlete1_back_double_bicep.jpg"
    result = analyze_xframe(IMAGE_NAME)
    
    if result:
        img = result['annotated_image']
        h, w = img.shape[:2]
        scale = 800 / h
        dim = (int(w * scale), 800)
        resized = cv2.resize(img, dim)
        cv2.imshow("X-Frame Analysis", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()