# WORK IS PROGRESS OVERALL STRUCTURE IS IN BUT I AM NOT HAPPY WITH THE PROCESS OF GETTING THE SCORE - FOR NOW GOTTA HUMBLE !
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    image_path: str = "athlete1_back_double_bicep.jpg"
    
    color_overall: tuple = (0, 255, 255)
    color_detail: tuple = (255, 0, 255)
    color_separation: tuple = (0, 165, 255)
    color_vascularity: tuple = (255, 100, 0)
    
    detail_kernel_size: int = 3
    edge_threshold_low: int = 50
    edge_threshold_high: int = 150
    vein_thickness_min: float = 2.0
    vein_thickness_max: float = 15.0
    
    weight_detail: float = 0.35
    weight_separation: float = 0.30
    weight_vascularity: float = 0.20
    weight_tightness: float = 0.15
    
    norm_detail: float = 10.0
    norm_separation: float = 50.0
    norm_vascularity: float = 20.0
    norm_tightness: float = 30.0


@dataclass
class ConditioningMetrics:
    detail: float
    separation: float
    vascularity: float
    tightness: float
    overall_score: float
    grade: str
    grade_color: tuple


class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_segmentation = mp.solutions.selfie_segmentation
    
    def extract_mask_and_landmarks(self, image):
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
                pose_results = pose.process(rgb)
                landmarks = pose_results.pose_landmarks
            
            with self.mp_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
                seg_results = segmenter.process(rgb)
                mask = (seg_results.segmentation_mask > 0.5).astype(np.uint8) * 255
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask, landmarks
        except Exception as e:
            logger.error(f"Error in pose analysis: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), None


class RegionExtractor:
    @staticmethod
    def get_analysis_regions(mask, landmarks, width, height):
        regions = {'full_body': mask}
        
        if landmarks is None:
            logger.warning("No landmarks detected. Using full body mask only.")
            return regions
        
        try:
            lm = landmarks.landmark
            l_sh = (int(lm[11].x * width), int(lm[11].y * height))
            r_sh = (int(lm[12].x * width), int(lm[12].y * height))
            l_hip = (int(lm[23].x * width), int(lm[23].y * height))
            r_hip = (int(lm[24].x * width), int(lm[24].y * height))
            l_knee = (int(lm[25].x * width), int(lm[25].y * height))
            r_knee = (int(lm[26].x * width), int(lm[26].y * height))
            
            shoulder_y = (l_sh[1] + r_sh[1]) // 2
            hip_y = (l_hip[1] + r_hip[1]) // 2
            knee_y = (l_knee[1] + r_knee[1]) // 2
            
            torso_len = max(hip_y - shoulder_y, 1)
            leg_len = max(knee_y - hip_y, 1)
            
            upper_back = np.zeros_like(mask)
            upper_start = max(0, shoulder_y)
            upper_end = min(height, shoulder_y + int(torso_len * 0.6))
            upper_back[upper_start:upper_end, :] = mask[upper_start:upper_end, :]
            regions['upper_back'] = upper_back
            
            lower_back = np.zeros_like(mask)
            lower_start = max(0, shoulder_y + int(torso_len * 0.5))
            lower_end = min(height, hip_y + int(torso_len * 0.1))
            lower_back[lower_start:lower_end, :] = mask[lower_start:lower_end, :]
            regions['lower_back'] = lower_back
            
            glute = np.zeros_like(mask)
            glute_start = max(0, hip_y - int(leg_len * 0.1))
            glute_end = min(height, hip_y + int(leg_len * 0.4))
            glute[glute_start:glute_end, :] = mask[glute_start:glute_end, :]
            regions['glutes'] = glute
            
            quad = np.zeros_like(mask)
            quad_start = max(0, hip_y)
            quad_end = min(height, knee_y - int(leg_len * 0.1))
            quad[quad_start:quad_end, :] = mask[quad_start:quad_end, :]
            regions['quads'] = quad
        except Exception as e:
            logger.error(f"Error extracting regions: {e}")
        
        return regions


class ConditioningAnalyzer:
    def __init__(self, config):
        self.config = config
    
    def analyze_muscle_detail(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        blur = cv2.GaussianBlur(masked, (self.config.detail_kernel_size, self.config.detail_kernel_size), 0)
        detail = cv2.subtract(masked, blur)
        
        muscle_px = masked[mask > 0]
        if len(muscle_px) < 100:
            return 0.0
        
        detail_px = detail[mask > 0]
        detail_var = np.var(detail_px.astype(float))
        mean_int = np.mean(muscle_px)
        
        if mean_int < 10:
            return 0.0
        
        return float(detail_var / mean_int)
    
    def analyze_muscle_separation(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)
        
        edges = cv2.Canny(enhanced, self.config.edge_threshold_low, self.config.edge_threshold_high)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        edge_px = np.sum(edges > 0)
        mask_area = np.sum(mask > 0)
        
        if mask_area < 100:
            return 0.0
        
        return float((edge_px / mask_area) * 1000)
    
    def analyze_vascularity(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        inverted = cv2.bitwise_not(masked)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blackhat = cv2.morphologyEx(inverted, cv2.MORPH_BLACKHAT, kernel)
        
        _, vein_map = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
        vein_map = cv2.bitwise_and(vein_map, vein_map, mask=mask)
        
        if np.sum(vein_map) == 0:
            return 0.0
        
        dist = cv2.distanceTransform(vein_map, cv2.DIST_L2, 3)
        valid = np.logical_and(dist >= self.config.vein_thickness_min, dist <= self.config.vein_thickness_max)
        
        vein_px = np.sum(valid)
        mask_area = np.sum(mask > 0)
        
        if mask_area < 100:
            return 0.0
        
        return float((vein_px / mask_area) * 1000)
    
    def analyze_skin_tightness(self, image, mask):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        sx = cv2.Sobel(masked, cv2.CV_64F, 1, 0, ksize=5)
        sy = cv2.Sobel(masked, cv2.CV_64F, 0, 1, ksize=5)
        grad = np.sqrt(sx**2 + sy**2)
        
        mask_px = mask > 0
        if np.sum(mask_px) < 100:
            return 0.0
        
        return float(np.mean(grad[mask_px]))
    
    def calculate_overall_score(self, detail, separation, vascularity, tightness):
        detail_n = min((detail / self.config.norm_detail) * 100, 100)
        sep_n = min((separation / self.config.norm_separation) * 100, 100)
        vasc_n = min((vascularity / self.config.norm_vascularity) * 100, 100)
        tight_n = min((tightness / self.config.norm_tightness) * 100, 100)
        
        overall = (detail_n * self.config.weight_detail +
                  sep_n * self.config.weight_separation +
                  vasc_n * self.config.weight_vascularity +
                  tight_n * self.config.weight_tightness)
        
        return overall
    
    def get_grade(self, score):
        if score >= 85:
            return "CONTEST READY", (0, 255, 0)
        elif score >= 70:
            return "EXCELLENT", (0, 200, 255)
        elif score >= 55:
            return "GOOD", (0, 165, 255)
        elif score >= 40:
            return "NEEDS WORK", (0, 100, 255)
        else:
            return "OFF-SEASON", (0, 0, 255)
    
    def analyze(self, image, mask):
        logger.info("Analyzing muscle detail...")
        detail = self.analyze_muscle_detail(image, mask)
        
        logger.info("Analyzing muscle separation...")
        separation = self.analyze_muscle_separation(image, mask)
        
        logger.info("Analyzing vascularity...")
        vascularity = self.analyze_vascularity(image, mask)
        
        logger.info("Analyzing skin tightness...")
        tightness = self.analyze_skin_tightness(image, mask)
        
        overall = self.calculate_overall_score(detail, separation, vascularity, tightness)
        grade, grade_color = self.get_grade(overall)
        
        return ConditioningMetrics(
            detail=detail,
            separation=separation,
            vascularity=vascularity,
            tightness=tightness,
            overall_score=overall,
            grade=grade,
            grade_color=grade_color
        )


class ResultVisualizer:
    def __init__(self, config):
        self.config = config
    
    def draw_overlay(self, image, metrics):
        img = image.copy()
        
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (400, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "CONDITIONING ANALYSIS", (20, 30), font, 0.6, (255, 255, 255), 2)
        
        cv2.line(img, (20, 45), (380, 45), (255, 255, 255), 1)
        
        cv2.putText(img, f"Detail: {metrics.detail:.2f}", (20, 75), font, 0.6, self.config.color_detail, 2)
        cv2.putText(img, f"Separation: {metrics.separation:.2f}", (20, 105), font, 0.6, self.config.color_separation, 2)
        cv2.putText(img, f"Vascularity: {metrics.vascularity:.2f}", (20, 135), font, 0.6, self.config.color_vascularity, 2)
        cv2.putText(img, f"Tightness: {metrics.tightness:.2f}", (20, 165), font, 0.6, (200, 200, 200), 2)
        
        cv2.line(img, (20, 185), (380, 185), (255, 255, 255), 1)
        
        cv2.putText(img, f"SCORE: {metrics.overall_score:.1f}/100", (20, 215), font, 0.7, self.config.color_overall, 2)
        cv2.putText(img, metrics.grade, (20, 250), font, 0.7, metrics.grade_color, 2)
        
        return img
    
    def display_results(self, image, max_height=800):
        h, w = image.shape[:2]
        
        if h > max_height:
            scale = max_height / h
            dim = (int(w * scale), max_height)
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        
        cv2.imshow("Conditioning Analysis", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def print_results(self, metrics):
        print("\n" + "=" * 60)
        print("CONDITIONING ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Muscle Detail:     {metrics.detail:.2f}")
        print(f"Muscle Separation: {metrics.separation:.2f}")
        print(f"Vascularity:       {metrics.vascularity:.2f}")
        print(f"Skin Tightness:    {metrics.tightness:.2f}")
        print("-" * 60)
        print(f"OVERALL SCORE:     {metrics.overall_score:.1f}/100")
        print(f"GRADE:             {metrics.grade}")
        print("=" * 60)


def main():
    config = AnalysisConfig()
    
    img_path = Path(config.image_path)
    if not img_path.exists():
        logger.error(f"Image not found: {config.image_path}")
        return
    
    image = cv2.imread(str(img_path))
    if image is None:
        logger.error(f"Failed to load image: {config.image_path}")
        return
    
    h, w = image.shape[:2]
    logger.info(f"Loaded image: {config.image_path} ({w}x{h})")
    
    logger.info("Extracting body segmentation...")
    pose_analyzer = PoseAnalyzer()
    mask, landmarks = pose_analyzer.extract_mask_and_landmarks(image)
    
    regions = RegionExtractor.get_analysis_regions(mask, landmarks, w, h)
    
    analysis_mask = regions.get('upper_back', regions['full_body'])
    if np.sum(analysis_mask) < 1000:
        logger.warning("Upper back region too small, using full body")
        analysis_mask = regions['full_body']
    
    analyzer = ConditioningAnalyzer(config)
    metrics = analyzer.analyze(image, analysis_mask)
    
    visualizer = ResultVisualizer(config)
    result_image = visualizer.draw_overlay(image, metrics)
    visualizer.print_results(metrics)
    visualizer.display_results(result_image)


if __name__ == "__main__":
    main()