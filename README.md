# Dual Camera Facial Recognition System with Raspberry Pi 5 and Hailo AI

A comprehensive facial recognition system using two cameras on Raspberry Pi 5 with Hailo-8L AI accelerator for real-time face detection and recognition.

## Features

- **Dual Camera Support**: Simultaneous processing from two camera modules
- **Real-time Face Detection**: Using RetinaFace model optimized for Hailo-8L
- **Face Recognition**: ArcFace embeddings for face matching and identification
- **Hardware Acceleration**: Leverages Hailo-8L AI accelerator for efficient inference
- **Multi-threading**: Concurrent processing of both camera streams
- **Face Database**: Persistent storage and management of known faces
- **Live Preview**: Real-time display of detection results with bounding boxes
- **Configuration Management**: Easy setup and configuration for different scenarios

## Hardware Requirements

- Raspberry Pi 5 (8GB recommended)
- Hailo AI Hat+ (with Hailo-8L accelerator)
- 2x Raspberry Pi Camera Module 3 or compatible cameras
- 2x Camera adapter cables for Raspberry Pi 5 (15-pin to 22-pin)
- MicroSD card (64GB or larger, Class 10)
- Power supply (5V 5A recommended)

## Software Requirements

- Raspberry Pi OS Bookworm (64-bit)
- Python 3.11+
- HailoRT 4.18.0+
- TAPPAS framework
- libcamera/rpicam-apps
- OpenCV 4.x
- Picamera2

## Installation

### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv git cmake build-essential
sudo apt install -y libcamera-dev libcamera-apps
sudo apt install -y python3-picamera2 python3-opencv
```

### 2. Hailo Setup

Follow the official Hailo installation guide to install:
- HailoRT PCIe driver
- TAPPAS framework
- Required models (RetinaFace, ArcFace)

### 3. Clone and Install

```bash
git clone <repository-url>
cd facial_recognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

### Camera Setup

The system supports dual cameras connected to CAM0 and CAM1 ports on Raspberry Pi 5.

1. **Physical Connection**: Connect cameras using appropriate adapter cables
2. **Software Configuration**: Cameras are automatically detected by libcamera

### Hailo Models

Required models for the system:
- `retinaface_mobilenet_v1.hef` - Face detection
- `arcface_mobilefacenet_2022-09.hef` - Face recognition embeddings

## Usage

### Basic Operation

```bash
# Run dual camera facial recognition
python3 dual_camera_face_recognition.py

# Run with specific configuration
python3 dual_camera_face_recognition.py --config config/dual_camera.yaml
```

### Face Database Management

```bash
# Add new face to database
python3 face_database.py --add --name "John Doe" --image path/to/image.jpg

# List registered faces
python3 face_database.py --list

# Remove face from database
python3 face_database.py --remove --name "John Doe"
```

## Architecture

The system consists of several key components:

1. **Camera Manager**: Handles dual camera initialization and frame capture
2. **Hailo Inference Engine**: Manages AI model loading and inference
3. **Face Detection Pipeline**: RetinaFace-based detection pipeline
4. **Face Recognition Pipeline**: ArcFace-based recognition pipeline
5. **Face Database**: SQLite-based storage for face embeddings
6. **Display Manager**: Real-time visualization of results
7. **Configuration Manager**: YAML-based configuration system

## Performance

- **Face Detection**: ~30 FPS per camera on Hailo-8L
- **Face Recognition**: ~15 FPS per camera with database lookup
- **Memory Usage**: ~2GB RAM typical usage
- **Latency**: <100ms end-to-end processing time

## Applications

- **Security Systems**: Multi-point access control
- **Retail Analytics**: Customer recognition and tracking
- **Smart Home**: Personalized user experiences
- **Event Management**: Automated check-in systems
- **Educational**: Attendance tracking systems

## Troubleshooting

Common issues and solutions are documented in the [troubleshooting guide](docs/troubleshooting.md).

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hailo Technologies for the AI acceleration platform
- Raspberry Pi Foundation for the hardware platform
- OpenCV community for computer vision tools
- Sanjoy G. for the original Hailo face recognition implementation