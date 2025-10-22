# Dual RTSP Camera Facial Recognition System

A comprehensive facial recognition system using two RTSP network cameras with DeGirum AI acceleration for real-time face detection and recognition on Raspberry Pi 5.

## Features

- **Dual RTSP Camera Support**: Simultaneous processing from two network/IP cameras
- **Real-time Face Detection**: Using DeGirum optimized models with high accuracy
- **Face Recognition**: DeGirum face recognition models for face matching and identification
- **Hardware Acceleration**: Leverages DeGirum AI accelerator for efficient inference
- **Multi-threading**: Concurrent processing of both camera streams
- **Face Database**: SQLite database with persistent storage of face embeddings
- **Face Tracking**: Kalman filter-based tracking across frames
- **Live Preview**: Real-time display of detection results with bounding boxes
- **Face Enrollment**: Easy enrollment system via camera or image files
- **Configuration Management**: YAML-based configuration system

## Project Structure

```
project/
├── main.py                 # Main application entry point
├── config.py              # Configuration management
├── enroll.py              # Face enrollment script
├── requirements.txt       # Python dependencies
├── config.yaml            # System configuration file
├── modules/
│   ├── detection.py       # Face detection using DeGirum
│   ├── tracking.py        # Face tracking across frames
│   ├── recognition.py     # Face recognition and matching
│   └── logging.py         # System logging utilities
└── data/                  # Database and embeddings storage
    ├── face_database.db   # SQLite database (auto-created)
    └── embeddings/        # Face embedding files (auto-created)
```

## Requirements

### Hardware
- Raspberry Pi 5
- DeGirum AI accelerator (Orca AI accelerator recommended)
- Two RTSP network cameras
- Minimum 4GB RAM recommended

### Software
- Python 3.8+
- OpenCV 4.5+
- DeGirum SDK
- PyYAML
- NumPy
- SQLite3

## Installation

1. **Clone or create the project directory:**
   ```bash
   mkdir dual_camera_face_recognition
   cd dual_camera_face_recognition
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install DeGirum SDK:**
   ```bash
   # Follow DeGirum installation instructions for your platform
   pip install degirum
   ```

4. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt update
   sudo apt install -y python3-opencv ffmpeg
   ```

5. **Configure your cameras:**
   Edit `config.yaml` and update the RTSP URLs for your cameras:
   ```yaml
   cameras:
     camera_0:
       rtsp_url: rtsp://admin:password@192.168.1.100:554/stream1
     camera_1:
       rtsp_url: rtsp://admin:password@192.168.1.101:554/stream1
   ```

## Usage

### 1. Face Enrollment

Before running the main system, you need to enroll faces in the database.

**Enroll from camera (webcam):**
```bash
python enroll.py --name "John Doe"
```

**Enroll from RTSP camera:**
```bash
python enroll.py --name "John Doe" --camera "rtsp://admin:password@192.168.1.100:554/stream1"
```

**Enroll from image file:**
```bash
python enroll.py --name "John Doe" --image "path/to/photo.jpg"
```

**List enrolled persons:**
```bash
python enroll.py --list
```

**Remove a person:**
```bash
python enroll.py --remove "John Doe"
```

### 2. Run the Main System

**Start the dual camera system:**
```bash
python main.py
```

**Run with custom configuration:**
```bash
python main.py --config custom_config.yaml
```

**Run without display (headless mode):**
```bash
python main.py --no-display
```

**Run with debug logging:**
```bash
python main.py --debug
```

### 3. System Controls

While the system is running:
- **Press 'q' or ESC**: Quit the application
- **Display shows**: 
  - Camera feeds side by side
  - Face detection bounding boxes
  - Person names and confidence scores
  - System statistics (FPS, detection counts)

## Configuration

The system uses a YAML configuration file (`config.yaml`) with the following sections:

### Cameras
```yaml
cameras:
  camera_0:
    enabled: true
    name: "Front Door Camera"
    rtsp_url: "rtsp://admin:password@192.168.1.100:554/stream1"
    connection:
      timeout: 10
      reconnect_attempts: 5
      buffer_size: 1
```

### DeGirum Models
```yaml
degirum:
  device:
    type: auto  # auto, cpu, gpu, orca
    device_id: 0
  face_detection:
    model_name: "mobilenet_v2_ssd_coco--300x300_quant_n2x_orca1_1"
    confidence_threshold: 0.6
    nms_threshold: 0.4
  face_recognition:
    model_name: "facenet_keras--160x160_quant_n2x_orca1_1"
    similarity_threshold: 0.7
```

### Database
```yaml
database:
  path: "data/face_database.db"
  embeddings_path: "data/embeddings/"
  max_faces_per_person: 10
  similarity_threshold: 0.7
```

## DeGirum Models

The system supports various DeGirum models:

### Face Detection Models
- `mobilenet_v2_ssd_coco--300x300_quant_n2x_orca1_1`
- `yolo_v5s_face_detection--640x640_quant_n2x_orca1_1`
- `retinaface_mobilenet_v1--1024x1024_quant_n2x_orca1_1`

### Face Recognition Models
- `facenet_keras--160x160_quant_n2x_orca1_1`
- `arcface_resnet50--112x112_quant_n2x_orca1_1`
- `sphereface_resnet50--112x112_quant_n2x_orca1_1`

## Performance Optimization

### For Raspberry Pi 5:
1. **Use hardware acceleration**: Ensure DeGirum Orca accelerator is properly installed
2. **Optimize camera settings**: Use appropriate resolution and frame rate
3. **Adjust buffer sizes**: Set camera buffer size to 1 for low latency
4. **Monitor system resources**: Check CPU and memory usage

### Camera Settings:
```yaml
cameras:
  camera_0:
    connection:
      buffer_size: 1        # Reduce latency
      timeout: 10           # Connection timeout
      reconnect_attempts: 5 # Auto-reconnect attempts
```

## Troubleshooting

### Common Issues:

1. **Camera connection failed:**
   - Check RTSP URL format
   - Verify camera credentials
   - Test camera with VLC or similar player

2. **DeGirum model loading failed:**
   - Ensure DeGirum SDK is properly installed
   - Check model names in configuration
   - Verify hardware accelerator connection

3. **Low FPS performance:**
   - Reduce camera resolution
   - Check system resources
   - Verify hardware acceleration is working

4. **Face detection not working:**
   - Adjust confidence threshold
   - Check lighting conditions
   - Verify camera focus and positioning

### Debug Mode:
```bash
python main.py --debug
```

This enables detailed logging to help identify issues.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RTSP Camera   │────│  Camera Thread   │────│  Frame Buffer   │
│   (Camera 0)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   RTSP Camera   │────│  Camera Thread   │─────────────┤
│   (Camera 1)    │    │                  │             │
└─────────────────┘    └──────────────────┘             │
                                                         │
                       ┌──────────────────┐             │
                       │ Processing Thread│◄────────────┘
                       │                  │
                       └─────────┬────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
         ┌─────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
         │ Detection  │  │ Recognition │  │  Tracking   │
         │ (DeGirum)  │  │ (DeGirum)   │  │ (Kalman)    │
         └────────────┘  └─────────────┘  └─────────────┘
                                 │
                         ┌──────▼──────┐
                         │  Database   │
                         │ (SQLite)    │
                         └─────────────┘
```

## License

This project is for educational and research purposes. Please ensure compliance with DeGirum SDK license terms and camera usage regulations.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Enable debug mode for detailed logs
3. Verify hardware and software requirements
4. Check DeGirum SDK documentation for model-specific issues