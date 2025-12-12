# AI Perception & Interaction Suite

A computer vision collection featuring real-time Augmented Reality analysis and holographic gesture control. This repository contains two core applications:
1.  **AR Object & Mood Recognition:** An intelligent HUD that analyzes emotions and detects objects.
2.  **Holographic Fluid Cube:** A physics-based particle simulation controlled entirely by hand gestures.

---

## 🚀 Features

### Module 1: AR Object & Mood Recognition
* **Real-Time Object Detection:** Uses **YOLOv8 Nano** to identify interactive objects (Person, Cup, Cell Phone, Laptop, Bottle) at 30+ FPS.
* **Geometric Mood Analysis:** Ignores pixel color to prevent lighting errors. Instead, it calculates vector geometry of 468 facial landmarks to detect Happy, Sad, Surprised, and Neutral states.
* **Privacy Mode:** An interactive AR button that triggers a real-time Gaussian Blur over the user's face to redact identity.
* **Dynamic HUD:** Text and UI elements automatically snap inside bounding boxes to prevent occlusion.

### Module 2: Holographic Fluid Cube (New!)
* **Gesture-Based Physics Engine:** A 3D wireframe cube filled with 1,000 fluid particles that react to simulated gravity and centrifugal force.
* **Bimanual Control System:**
    * **Resize (Setup Mode):** Pinching with the **Left Hand** scales the cube (active only when the Right Hand is hidden).
    * **Spin & Tilt (Play Mode):** Using **Both Hands**, the distance between index fingers controls spin speed, while hand height controls the cube's tilt.
* **Fluid Dynamics:** Particles feature "anti-clumping" logic, wall collision bounce, and rotational gravity vectors to simulate liquid sloshing inside a moving container.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Computer Vision:** OpenCV (`cv2`), MediaPipe (Hands & Face Mesh), Ultralytics YOLOv8
* **Graphics & Simulation:** Pygame, PyOpenGL (OpenGL ES)
* **Math:** NumPy (Vector calculations)

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/yourusername/ai-perception-suite.git](https://github.com/yourusername/ai-perception-suite.git)
    cd ai-perception-suite
    ```

2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Ensure your `requirements.txt` includes:*
    ```text
    opencv-python
    numpy
    ultralytics
    mediapipe
    pygame
    PyOpenGL
    PyOpenGL_accelerate
    ```

---
##
Controls: Click the "PRIVACY" button on-screen to blur face. Press q to quit.
Running Module 2: Fluid Cube Controller
Bash
python fluid_cube.py
Resize Mode: Hide your right hand. Show your Left Hand and pinch your Index & Thumb to shrink/grow the cube.
Physics Mode: Show Both Hands.
Spin: Move index fingers apart (Distance = Speed).
Tilt: Move hands up or down (Height = Pitch).
Quit: Press q to exit.

🧠 How It Works1. The "Shadow Smile" Solution (Mood Module)Standard emotion AI fails in bad lighting.
This project solves that by calculating the Smile Curve:
$$Curve = Y_{corners} - Y_{center}$$
If Curve < -0.01: Corners are physically higher → Happy.
If Curve > 0.005: Corners are lower → Sad.This math holds true regardless of shadows or skin tone.

2. Rotational Gravity (Fluid Module)
To make particles stay "down" inside a spinning cube, the system cannot use static gravity ($0, -1, 0$). 
Instead, it calculates an Inverse Rotation Matrix.

If the cube tilts $45^\circ$, the gravity vector rotates $-45

^\circ$.$$G_{local} = R_{yaw}^{-1} \cdot R_{pitch}^{-1} \cdot G_{world}$$

This ensures that when you tilt the cube, the water flows to the new "bottom" corner naturally.

🔮 Future Improvements

Interaction Fusion: Using the "Cup" detection from Module 1 to "pour" the fluid particles from Module 2.

3D Anchoring: Integrating OpenGL to anchor virtual hats/glasses to MediaPipe landmarks.

Mobile Port: Optimizing the pipeline for Android/iOS using TensorFlow Lite.

📝 LicenseThis project is created for academic coursework (2024/2025).

## 🎮 Usage

### Running Module 1: Mood & Object AR
```bash
python ar_project.py



