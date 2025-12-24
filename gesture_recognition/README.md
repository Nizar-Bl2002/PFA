# Gesture Recognition Project

A real-time hand gesture recognition system using Mediapipe and OpenCV.

## Features

Detects and prints the following gestures to the terminal:
- **Index Finger Up**: Prints `up`
- **Index Finger Down**: Prints `down`
- **Index Finger Left**: Prints `left`
- **Index Finger Right**: Prints `right`
- **Open Hand**: Prints `off`
- **Closed Hand**: Prints `on`

## Installation

1.  Navigate to the `gesture_recognition` directory.
2.  (Optional) Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install mediapipe==0.10.14 opencv-python
    ```

## Usage

Run the main script:
```bash
python main.py
```

- A window will open showing your webcam feed with hand landmarks.
- Perform the gestures to see the output in your terminal.
- Press **'q'** to exit the application.
