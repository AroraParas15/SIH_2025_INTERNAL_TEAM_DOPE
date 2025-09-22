## Technology Stack

This project leverages the following technologies:

**Python**: We chose this language for its simplicity and extensive libraries for image processing and AI, making development quick and efficient.

**OpenCV**: This was essential for our core functionality, as it provided robust tools for image processing, such as detecting sand grain shapes and boundaries.

**NumPy**: We used this library for efficient numerical operations and data manipulation on the large datasets generated from our image analysis.

**Matplotlib**: We chose this tool to create visual representations of our data, allowing us to generate graphs and charts to display the distribution of sand grain sizes.

## Key Features

- Automatic Tray Detection: Accurately detects the tray boundary in the image for reliable scale calibration.

- Advanced Image Enhancement: Applies denoising, contrast adjustment, and CLAHE to handle noisy ESP32-CAM images.

- Grain Segmentation and Separation: Uses adaptive thresholding, morphological operations, and watershed algorithm to isolate and separate touching sand grains.

- Shape and Size Analysis: Extracts multiple metrics such as diameter, circularity, aspect ratio, solidity, and generates detailed statistics.

- Data Export and Visualization: Produces overlays, histograms, per-grain CSV data, and summary statistics for easy analysis.

- Batch Processing Support: Can process multiple images at once and generate a summary CSV of results.

## Local Setup Instructions (Write for both windows and macos)

Follow these steps to run the project locally
For Windows:
**Open Command Prompt**: Search for Command Prompt in your Start Menu.

**Navigate to the Project Folder:** Use the cd command to move into the directory where you saved your project files. For example:
cd C:\Users\YourUsername\Desktop\SIH_2025_Internal_Round_Submission

**Install Required Libraries**: You can install the most common libraries for your project by running these commands:
pip install opencv-python
pip install numpy

**Run the Project**: Now, you can run your project's main script. Replace your_main_file.py with the actual name of your project's main Python file.
python your_main_file.py


For MacOS:
**Open Terminal**: Find the Terminal app in Applications > Utilities.

**Navigate to the Project Folder**: Use the cd command to go to your project's folder. For example:
cd ~/Documents/SIH_2025_Internal_Round_Submission

**Install Required Libraries**: Use pip3 to install the common libraries your project uses.
pip3 install opencv-python
pip3 install numpy

**Run the Project:** Finally, run the main script for your project. Remember to use the correct file name.
python3 your_main_file.py


**SOURCE CODE** :

import cv2
import numpy as np
import math
from scipy import ndimage
from skimage import morphology, measure, segmentation
import matplotlib.pyplot as plt
import os
import csv
import json
from pathlib import Path
import random

class GrainAnalyzer:
    def __init__(self, config_path=None):
        # Default configuration
        self.config = {
            "tray_width_mm": 50.0,
            "min_diam_mm": 0.1,
            "max_diam_mm": 2.5,
            "min_circularity": 0.4,
            "min_solidity": 0.8,
            "max_aspect_ratio": 2.2,
            "morph_kernel_size": 3,
            "adaptive_block_size": 51,
            "denoise_strength": 10,
            "clahe_clip_limit": 2.0,
            "watershed_marker_threshold": 0.5,
            "contrast_alpha": 1.2,
            "contrast_beta": 10,
            "border_clear_margin": 10
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
                
        # Pre-calculate kernel for morphological operations
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.config["morph_kernel_size"], self.config["morph_kernel_size"])
        )
    
    def detect_tray(self, image):
        """Robust tray detection with multiple fallback strategies"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Strategy 1: Look for prominent edges
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Strategy 2: Use the whole image as fallback
            h, w = gray.shape
            return np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), w
            
        # Find the largest contour that is reasonably rectangular
        best_cnt = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < gray.size * 0.1:  # Too small to be the tray
                continue
                
            # Approximate contour to polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Calculate rectangularity score
            if len(approx) == 4:  # It's a quadrilateral
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                score = extent * (area / gray.size)
                
                if score > best_score:
                    best_score = score
                    best_cnt = cnt
        
        # Fallback to largest contour if no good quadrilateral found
        if best_cnt is None:
            best_cnt = max(contours, key=cv2.contourArea)
            
        # Get bounding box for scale calculation
        x, y, w, h = cv2.boundingRect(best_cnt)
        return best_cnt, max(w, h)
    
    def enhance_image(self, image):
        """Specialized enhancement for ESP32-CAM images"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise (important for ESP32-CAM images)
        denoised = cv2.fastNlMeansDenoising(gray, None, self.config["denoise_strength"], 7, 21)
        
        # Contrast enhancement
        contrasted = cv2.convertScaleAbs(denoised, alpha=self.config["contrast_alpha"], 
                                        beta=self.config["contrast_beta"])
        
        # Background correction
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        bg = cv2.morphologyEx(contrasted, cv2.MORPH_CLOSE, kernel_bg)
        bg = cv2.medianBlur(bg, 21)
        corrected = cv2.subtract(bg, contrasted)
        
        # CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=self.config["clahe_clip_limit"], 
                               tileGridSize=(8, 8))
        enhanced = clahe.apply(corrected)
        
        return enhanced
    
    def segment_grains(self, enhanced, tray_mask):
        """Optimized segmentation for grain detection"""
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.config["adaptive_block_size"], 2
        )
        
        # Apply tray mask
        thresh = cv2.bitwise_and(thresh, tray_mask)
        
        # Morphological cleaning
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel, iterations=2)
        
        # Clear border artifacts (common in ESP32-CAM images)
        border_cleared = segmentation.clear_border(cleaned, 
                                                 buffer_size=self.config["border_clear_margin"])
        
        return border_cleared
    
    def separate_touching_grains(self, binary_image):
        """Improved watershed implementation"""
        # Distance transform
        dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        # Find sure foreground with higher threshold to reduce over-segmentation
        _, sure_fg = cv2.threshold(
            dist, 
            self.config["watershed_marker_threshold"] * dist.max(), 
            255, 
            0
        )
        sure_fg = np.uint8(sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Apply watershed
        markers = markers + 1
        unknown = cv2.subtract(binary_image, sure_fg)
        markers[unknown == 255] = 0
        
        markers_ws = cv2.watershed(
            cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), 
            markers
        )
        
        return markers_ws
    
    def analyze_grains(self, markers, mm_per_px):
        """Comprehensive grain analysis with multiple shape metrics"""
        diameters_mm = []
        grain_properties = []
        
        for m in range(2, markers.max() + 1):  # Start from 2 (background is 1)
            mask = np.uint8(markers == m)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            cnt = contours[0]
            area_px = cv2.contourArea(cnt)
            
            # Skip very small areas
            if area_px < 10:
                continue
                
            # Calculate equivalent diameter
            eq_d_px = math.sqrt(4.0 * area_px / math.pi)
            eq_d_mm = eq_d_px * mm_per_px
            
            # Size filtering
            if not (self.config["min_diam_mm"] <= eq_d_mm <= self.config["max_diam_mm"]):
                continue
                
            # Calculate shape metrics
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * math.pi * (area_px / (perimeter * perimeter))
            
            # Aspect ratio
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area_px / hull_area if hull_area > 0 else 0
            
            # Apply filters
            if (circularity < self.config["min_circularity"] or 
                solidity < self.config["min_solidity"] or 
                aspect_ratio > self.config["max_aspect_ratio"]):
                continue
                
            diameters_mm.append(eq_d_mm)
            
            # Store detailed properties
            grain_properties.append({
                "diameter_mm": eq_d_mm,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
                "area_px": area_px,
                "contour": cnt
            })
            
        return diameters_mm, grain_properties
    
    def process_image(self, image_path, output_dir):
        """Main processing pipeline"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Create output directory
        image_name = Path(image_path).stem
        result_dir = Path(output_dir) / image_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect tray and calculate scale
        tray_contour, tray_width_px = self.detect_tray(image)
        mm_per_px = self.config["tray_width_mm"] / tray_width_px
        
        # Create tray mask
        tray_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(tray_mask, [tray_contour], -1, 255, -1)
        
        # Enhance image
        enhanced = self.enhance_image(image)
        
        # Segment grains
        segmented = self.segment_grains(enhanced, tray_mask)
        
        # Separate touching grains
        markers = self.separate_touching_grains(segmented)
        
        # Analyze grains
        diameters_mm, grain_properties = self.analyze_grains(markers, mm_per_px)
        
        # Generate results
        results = self.generate_results(image, diameters_mm, grain_properties, 
                                      tray_contour, result_dir)
        
        return results
    
    def generate_results(self, image, diameters_mm, grain_properties, 
                        tray_contour, result_dir):
        """Generate output visualizations and data"""
        # Create overlay image
        overlay = image.copy()
        cv2.drawContours(overlay, [tray_contour], -1, (0, 255, 0), 2)
        
        # Draw grains with random colors
        for prop in grain_properties:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(overlay, [prop["contour"]], -1, color, -1)
            
        # Calculate statistics
        stats = {
            "image": result_dir.name,
            "total_grains": len(diameters_mm),
            "mean_diam_mm": np.mean(diameters_mm) if diameters_mm else 0,
            "std_dev_mm": np.std(diameters_mm) if diameters_mm else 0,
            "min_diam_mm": np.min(diameters_mm) if diameters_mm else 0,
            "max_diam_mm": np.max(diameters_mm) if diameters_mm else 0,
            "D10": np.percentile(diameters_mm, 10) if diameters_mm else 0,
            "D50": np.percentile(diameters_mm, 50) if diameters_mm else 0,
            "D90": np.percentile(diameters_mm, 90) if diameters_mm else 0,
            "scale_mm_per_px": self.config["tray_width_mm"] / cv2.boundingRect(tray_contour)[2]
        }
        
        # Save results
        cv2.imwrite(str(result_dir / "overlay.png"), overlay)
        
        # Save histogram
        if diameters_mm:
            plt.figure(figsize=(10, 6))
            plt.hist(diameters_mm, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel("Grain Diameter (mm)")
            plt.ylabel("Count")
            plt.title(f"Grain Size Distribution: {result_dir.name}\nTotal grains: {len(diameters_mm)}")
            plt.grid(True, alpha=0.3)
            plt.savefig(str(result_dir / "histogram.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save statistics
        with open(str(result_dir / "stats.txt"), "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
                
        # Save detailed grain data
        with open(str(result_dir / "grain_data.csv"), "w", newline="") as f:
            if grain_properties:
                writer = csv.DictWriter(f, fieldnames=grain_properties[0].keys())
                writer.writeheader()
                writer.writerows(grain_properties)
                
        return stats

def batch_process(input_dir, output_dir, config_path=None):
    """Process all images in a directory"""
    analyzer = GrainAnalyzer(config_path)
    all_stats = []
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    for i, image_file in enumerate(image_files):
        print(f"Processing {image_file.name} ({i+1}/{len(image_files)})")
        
        try:
            stats = analyzer.process_image(str(image_file), output_dir)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
    
    # Save summary
    if all_stats:
        with open(output_path / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
            writer.writeheader()
            writer.writerows(all_stats)
            
    print(f"Processing complete. Results saved to {output_dir}")
if __name__ == "__main__":
    # Hardcoded paths for your setup
    IMAGES_DIR = r"D:\sand_grain_project\images"
    RESULTS_DIR = r"D:\sand_grain_project\results"
    CONFIG_PATH = None  # Or specify a config file if you have one
    
    batch_process(IMAGES_DIR, RESULTS_DIR, CONFIG_PATH)


