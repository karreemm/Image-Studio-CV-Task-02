# Vision Lab

VisionLab is a powerful computer vision desktop application built with PyQt5 that provides advanced image processing capabilities including active contour segmentation, edge detection, and geometric shape recognition. The application features an intuitive graphical interface for interactive computer vision tasks and real-time parameter adjustment.

## ✨ Features

### 🎯 **Active Contour (Snake) Algorithm**
- **Interactive Contour Drawing**: Draw initial contours using mouse interaction
- **Multiple Drawing Modes**: Free drawing, rectangle, and circle modes
- **Greedy Snake Algorithm**: Advanced active contour optimization
- **Real-time Parameters**: Adjustable alpha (elasticity), beta (curvature), gamma (external energy), and window size
- **Contour Analytics**: Automatic calculation of perimeter, area, and chain code

### 🔍 **Canny Edge Detection**
- **Gaussian Smoothing**: Configurable sigma parameter for noise reduction
- **Dual Thresholding**: Customizable low and high threshold values
- **Edge Tracking**: Hysteresis-based edge linking
- **Real-time Processing**: Live preview with parameter adjustment

### 📐 **Geometric Shape Detection**
- **Hough Transform Implementation**: For detecting lines, circles, and ellipses
- **Voting Threshold Control**: Adjustable detection sensitivity for each shape type
- **Overlay Visualization**: Detected shapes overlaid on original images
- **Multi-shape Detection**: Simultaneous detection of multiple geometric primitives

### 🖱️ **Interactive User Interface**
- **Modern Dark Theme**: Professional dark UI with blue accents
- **Real-time Sliders**: Instant parameter adjustment with visual feedback
- **Image Loading**: Support for multiple image formats (JPG, PNG, BMP, etc.)
- **Split View**: Side-by-side input and output image comparison

## 🖼️ **Preview**

![GUI Active Contour Screenshot](https://github.com/Youssef-Abo-El-Ela/ImageStudio-2/blob/main/assets/Active%20Vontour.png)

*VisionLab-CV interface showing active contour segmentation with real-time parameter controls*

## 🚀 **Installation**

### Prerequisites

- **Python 3.7+** 
- **Operating System**: Windows, macOS, or Linux

### Required Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install PyQt5 opencv-python numpy
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyQt5 | 5.15+ | GUI framework and user interface |
| OpenCV | 4.0+ | Computer vision and image processing |
| NumPy | 1.19+ | Numerical computations and array operations |

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Youssef-Abo-El-Ela/ImageStudio-2.git
   cd ImageStudio-2
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python main.py
   ```

## 💡 **Usage Guide**

### Getting Started

1. **Launch Application**: Run `python main.py` to start VisionLab-CV
2. **Load Image**: Click "Browse" to select an input image (JPG, PNG, BMP, etc.)
3. **Choose Processing Mode**: Select from Active Contour or Edge Detection tabs

### Active Contour (Snake) Workflow

1. **Draw Initial Contour**:
   - Select drawing mode: Free, Rectangle, or Circle
   - Click and drag on the input image to create initial contour
   - Use predefined shapes for regular contours

2. **Configure Parameters**:
   - **Alpha (α)**: Controls contour elasticity (0.1-2.0)
   - **Beta (β)**: Controls contour curvature smoothness (0.1-2.0)  
   - **Gamma (γ)**: Controls external energy influence (0.1-2.0)
   - **Window Size**: Search window for optimization (5-20 pixels)

3. **Apply Snake Algorithm**: Click "Apply Snake" to optimize the contour
4. **View Results**: Analyze contour metrics (perimeter, area, chain code)

### Edge Detection Workflow

1. **Set Parameters**:
   - **Sigma**: Gaussian smoothing intensity (0.5-5.0)
   - **Low Threshold**: Lower edge threshold (10-100)
   - **High Threshold**: Upper edge threshold (50-300)

2. **Shape Detection** (Optional):
   - Enable line, circle, or ellipse detection
   - Adjust voting thresholds for each shape type (1-100%)

3. **Apply Processing**: Click "Apply Canny" to process the image
4. **Analyze Results**: View detected edges and geometric shapes

## **Contributors** <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/karreemm" target="_black">
    <img src="https://avatars.githubusercontent.com/u/116344832?v=4" width="150px;" alt="Kareem Abdel Nabi"/>
    <br />
    <sub><b>Kareem Abdel nabi</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/Youssef-Abo-El-Ela" target="_black">
    <img src="https://avatars.githubusercontent.com/u/125592387?v=4" width="150px;" alt="Youssef Aboelela"/>
    <br />
    <sub><b>Youssef Aboelela</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/aliyounis33" target="_black">
    <img src="https://avatars.githubusercontent.com/u/125222093?v=4" width="150px;" alt="Ali Younis"/>
    <br />
    <sub><b>Ali Younis</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/louai111" target="_black">
    <img src="https://avatars.githubusercontent.com/u/79408256?v=4" width="150px;" alt="Louai Khaled"/>
    <br />
    <sub><b>Louai Khaled</b></sub></a>
    </td>
      </tr>
</table>
