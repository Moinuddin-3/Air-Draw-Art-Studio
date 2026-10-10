# 🎨 Air Draw Controller

**Draw, paint, and create shapes in the air using hand gestures and your webcam!**

Air Draw Controller is a computer vision-based drawing and painting application built with Python, OpenCV, and MediaPipe. It lets users draw in the air without directly touching the mouse or keyboard for drawing operations.

## ✨ Features

- 🖐️ **Real-Time Hand Tracking** — Detect and track hand movements through a webcam.
- 🎨 **Multiple Drawing Tools** — Pencil, Pen, Brush, Marker, Highlighter, and Eraser.
- 🔷 **2,000-Shape Library** — Includes 2D shapes, 3D-style solids, wireframes, models, patterns, curves, and more.
- 🌈 **Color and Style Options** — Customize colors, brush sizes, fill modes, and art styles.
- ↩️ **Undo and Redo** — Reverse or restore drawing actions.
- 🧹 **Clear Canvas** — Start a fresh drawing.
- 💾 **Save Artwork** — Save drawings as PNG images.
- 🤲 **Gesture-Based Controls** — Use hand gestures to draw and interact with controls.
- 🧊 **3D Rotation Controls** — Rotate supported 3D-style shapes along the X, Y, and Z axes.

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## 📋 Requirements

- Python 3.13 or a compatible Python version
- A webcam
- Windows, or another supported environment configured for the required libraries

## 🚀 Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/Moinuddin-3/Air-Draw-Art-Studio.git
```

**2. Open the project folder**

```bash
cd Air-Draw-Art-Studio
```

**3. Install dependencies**

```bash
pip install opencv-python mediapipe numpy
```

**4. Run the application**

```bash
python air_draw_controller.py
```

Allow webcam access and ensure your hands are visible to the camera.

## 🖐️ Hand Gestures

- **Right hand:** Draw and create shapes.
- **Left hand:** Select and control tools.
- **Pinch gesture:** Activate a control when positioned over the panel.
- **Two hands:** Use supported two-hand drawing and shape-sizing controls.

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| U | Undo |
| R | Redo |
| C | Clear canvas |
| S | Save drawing |
| P | Toggle paper mode |
| F | Change fill mode |
| A | Change art style |
| T | Change tool |
| G | Open shape palette |
| V | Reset 3D rotation |
| Q / Esc | Quit application |

Additional shortcuts are displayed when the application starts.

## 📂 Project Structure

```text
Air-Draw-Art-Studio/
├── air_draw_controller.py
├── hand_landmarker.task
├── airdraw_saves/
└── README.md
```

- `air_draw_controller.py` — Main application.
- `hand_landmarker.task` — Hand landmark model used by the application.
- `airdraw_saves/` — Saved artwork.

## 🎯 Project Objective

The goal of this project is to explore computer vision and gesture-based interaction through an interactive drawing application. It provides a creative way to draw, paint, and experiment with shapes using hand movements.

## 👨‍💻 Developer

**MD Moinuddin**

- GitHub: [Moinuddin-3](https://github.com/Moinuddin-3)
- Project Repository: [Air Draw Controller](https://github.com/Moinuddin-3/Air-Draw-Art-Studio)

---

⭐ If you find this project interesting, consider giving the repository a star!
