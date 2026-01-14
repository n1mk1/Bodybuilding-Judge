# AI-Powered Bodybuilding Competition Judge 💪

An automated judging system that uses computer vision and machine learning to evaluate bodybuilding competitors across multiple scoring criteria. The system analyzes physique photos to provide objective, quantifiable assessments similar to professional competition judging standards.

<p align="center">
  <img src="https://github.com/n1mk1/Bodybuilding-Judge/blob/main/bd_judge/ui_image.png" width="1000"/>
  <img src="https://github.com/n1mk1/Bodybuilding-Judge/blob/main/bd_judge/ui_image2.png" width="1000"/>
  <img src="https://github.com/user-attachments/assets/e21db7aa-70e1-4257-9400-34b1dab2950a" width="200"/>
  <img src="https://github.com/user-attachments/assets/da18788a-51f1-4a44-8815-4bdc7a995ff4" width="200"/>
  <img src="https://github.com/user-attachments/assets/9a3564c4-d2c1-47cf-a63a-04f4032ca7a1" width="200"/>
  <img src="https://github.com/user-attachments/assets/d4c28348-32cf-4fe7-bc03-63e5ed7ca8ec" width="200"/>
</p>

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-orange.svg)
![License](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-blue.svg)

## Features

### 1. **Automated Pose Classification**
- Identifies and categorizes mandatory bodybuilding poses (front double bicep, back double bicep, etc.)
- Automatically organizes athlete images by pose type
- Supports multi-athlete comparison workflows
- MediaPipe-based skeletal tracking for accurate pose detection

### 2. **X-Frame Analysis**
- Measures shoulder width, lat spread, waist tightness, and quad development
- Calculates X-frame ratio: `(Lat Width + Quad Width) / Waist Width`
- Generates mass score: `(Shoulder + Lat + Quad) / Waist`
- Provides pixel-based measurements with color-coded visual overlays

### 3. **Conditioning Assessment**
- **Muscle Detail**: Analyzes striations and muscle fiber visibility
- **Muscle Separation**: Evaluates definition between muscle groups
- **Vascularity**: Detects and quantifies vein prominence
- **Skin Tightness**: Measures skin adherence to muscle
- Generates composite conditioning scores (0-100) with contest-readiness grades: [WORK IN PROGRESS. CURRENTLY MEME]

### 4. **Visual Feedback System**
- Annotated images with measurement lines and metrics
- Color-coded scoring indicators
- Real-time visual overlays showing analysis regions
- Detailed numerical breakdowns for each criterion

## Getting Started

### Prerequisites

```bash
Python 3.11+
pip
```

### Installation

1. **Clone the repository**
```bash
git clone [https://github.com//bodybuilding-judge.git](https://github.com/n1mk1/Bodybuilding-Judge.git)
cd bodybuilding-judge
```

2. **Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install opencv-python numpy mediapipe
```

## Usage

### Pose Classification & Organization

Automatically classify and organize athlete images:

```python
from poseClassifier import ImageOrganizer

organizer = ImageOrganizer(working_dir="./images")
result = organizer.organize_poses(overwrite=False)

print(f"Successfully organized {result['renamed_count']} images")
print(f"Athlete 1: {result['athlete1_count']} poses")
print(f"Athlete 2: {result['athlete2_count']} poses")
```

### X-Frame Analysis

Analyze proportions and frame measurements:

```python
from xframe import analyze_xframe

result = analyze_xframe("athlete1_back_double_bicep.jpg")

if result:
    print(f"X-Frame Ratio: {result['metrics']['x_frame']:.2f}")
    print(f"Mass Score: {result['metrics']['mass']:.2f}")
    print(f"Shoulder: {result['metrics']['shoulder']} px")
    print(f"Lat: {result['metrics']['lat']} px")
    print(f"Waist: {result['metrics']['waist']} px")
    print(f"Quad: {result['metrics']['quad']} px")
```

### Conditioning Analysis

Evaluate muscle conditioning and contest readiness:

```python
from conditioning import AnalysisConfig, PoseAnalyzer, ConditioningAnalyzer, ResultVisualizer
import cv2

# Configure analysis
config = AnalysisConfig(image_path="athlete1_back_double_bicep.jpg")

# Load and analyze image
image = cv2.imread(config.image_path)
pose_analyzer = PoseAnalyzer()
mask, landmarks = pose_analyzer.extract_mask_and_landmarks(image)

# Run conditioning analysis
analyzer = ConditioningAnalyzer(config)
metrics = analyzer.analyze(image, mask)

# Display results
visualizer = ResultVisualizer(config)
visualizer.print_results(metrics)
result_image = visualizer.draw_overlay(image, metrics)
visualizer.display_results(result_image)
```

## Project Structure

```
bodybuilding-judge/
├── UI/UI.py
├── poseClassifier.py      # Pose detection and image organization
├── xframe.py              # Proportions and frame analysis
├── conditioning.py        # Muscle conditioning assessment
├── README.md              # Project documentation
├── .gitignore            # Git ignore rules
└── images --> Remove all current test images. Input your own images for Front double bicep and Back double bicep for both athletes i.e 4 images
```

## 🔧 Configuration

### Conditioning Analysis Parameters

Customize analysis settings in `conditioning.py`:

```python
@dataclass
class AnalysisConfig:
    # Color scheme for visualization
    color_overall: tuple = (0, 255, 255)      # Cyan
    color_detail: tuple = (255, 0, 255)       # Magenta
    color_separation: tuple = (0, 165, 255)   # Orange
    color_vascularity: tuple = (255, 100, 0)  # Blue
    
    # Analysis parameters
    detail_kernel_size: int = 3
    edge_threshold_low: int = 50
    edge_threshold_high: int = 150
    vein_thickness_min: float = 2.0
    vein_thickness_max: float = 15.0
    
    # Scoring weights
    weight_detail: float = 0.35
    weight_separation: float = 0.30
    weight_vascularity: float = 0.20
    weight_tightness: float = 0.15
```

## How It Works

### Pose Detection
1. MediaPipe Pose extracts 33 skeletal landmarks
2. Angle calculations determine arm positions
3. Visibility checks confirm front/back orientation
4. Images automatically sorted by athlete and pose

### X-Frame Measurement
1. Body segmentation isolates the athlete from background
2. Arms removed from torso mask to prevent measurement interference
3. Horizontal scans at anatomical landmarks find body edges
4. Ratios calculated to quantify aesthetic taper

### Conditioning Scoring
1. **Detail**: Gaussian blur subtraction reveals muscle texture variance
2. **Separation**: CLAHE enhancement + Canny edge detection quantifies muscle boundaries
3. **Vascularity**: Morphological blackhat transform + distance transform identifies veins
4. **Tightness**: Sobel gradient magnitude measures skin adherence
5. Weighted combination produces final 0-100 score [NOT OPTIMAL]

## Use Cases

- **Competition Organizers**: Consistent, bias-free preliminary scoring
- **Athletes**: Objective progress tracking during prep phases
- **Coaches**: Data-driven client assessment and feedback
- **Judge Training**: Educational tool for learning scoring criteria

## Current Limitations

- Lighting conditions significantly impact conditioning scores
- Limited to double bicep poses (front/back)
- Requires high-resolution images for accurate measurements
- Background removal works best with solid, contrasting backgrounds
- Conditioning algorithm still under refinement (see WIP note in code)

## Roadmap

- [ ] Add support for additional mandatory poses (side chest, side tricep, etc.)
- [ ] Implement side-by-side athlete comparison mode
- [ ] Machine learning model for learned scoring vs. rule-based
- [ ] Web interface for easy upload and analysis
- [ ] Historical tracking and progress visualization
- [ ] Export detailed PDF reports with all metrics
- [ ] Calibration system for different lighting conditions

## License Creative Commons BY-NC-SA 4.0 license
