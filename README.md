# Dual RTSP Camera Facial Recognition System with Raspberry Pi 5 and DeGirum

A comprehensive facial recognition system using two RTSP cameras on Raspberry Pi 5 with DeGirum AI accelerator for real-time face detection and recognition.

## Features

- **Dual RTSP Camera Support**: Simultaneous processing from two network/IP cameras
- **Real-time Face Detection**: Using DeGirum optimized models
- **Face Recognition**: DeGirum face recognition models for face matching and identification
- **Hardware Acceleration**: Leverages DeGirum AI accelerator for efficient inference
- **Multi-threading**: Concurrent processing of both camera streams
- **Face Database**: Persistent storage and management of known faces
- **Live Preview**: Real-time display of detection results with bounding boxes
- **Configuration Management**: Easy setup and configuration for different RTSP sources
- **Network Resilience**: Automatic reconnection and error handling for RTSP streams

## Hardware Requirements

- Raspberry Pi 5 (8GB recommended)
- DeGirum AI accelerator (USB or PCIe)
- 2x IP/Network cameras with RTSP support
- Ethernet connection or Wi-Fi for camera access
- MicroSD card (64GB or larger, Class 10)
- Power supply (5V 5A recommended)

## Software Requirements

- Raspberry Pi OS Bookworm (64-bit)
- Python 3.11+
- DeGirum Runtime and SDK
- OpenCV 4.x with GStreamer support
- FFmpeg with RTSP support

## Installation

### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv git cmake build-essential
sudo apt install -y python3-opencv ffmpeg
sudo apt install -y gstreamer1.0-plugins-base gstreamer1.0-plugins-good
sudo apt install -y gstreamer1.0-plugins-bad gstreamer1.0-libav
```

### 2. DeGirum Setup

Follow the official DeGirum installation guide:

```bash
# Install DeGirum SDK
pip install degirum

# Verify installation
python3 -c "import degirum as dg; print(dg.__version__)"
```

### 3. Clone and Install

```bash
git clone <repository-url>
cd facial_recognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

### RTSP Camera Setup

Configure your IP cameras with RTSP streams in the configuration file:

```yaml
cameras:
  camera_0:
    rtsp_url: "rtsp://user:password@192.168.1.100:554/stream1"
    name: "Front Door Camera"
    enabled: true
    
  camera_1:
    rtsp_url: "rtsp://user:password@192.168.1.101:554/stream1"
    name: "Back Door Camera"
    enabled: true
```

### DeGirum Models

The system uses DeGirum's pre-trained models:
- Face detection model from DeGirum model zoo
- Face recognition/embedding model from DeGirum model zoo

## Usage

### Basic Operation

```bash
# Run dual RTSP camera facial recognition
python3 dual_rtsp_face_recognition.py

# Run with specific configuration
python3 dual_rtsp_face_recognition.py --config config/dual_rtsp.yaml
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

### RTSP Stream Testing

```bash
# Test RTSP connection
python3 tools/test_rtsp.py --url "rtsp://user:password@192.168.1.100:554/stream1"

# View RTSP stream
ffplay "rtsp://user:password@192.168.1.100:554/stream1"
```

## Architecture

The system consists of several key components:

1. **RTSP Stream Manager**: Handles RTSP stream connections and frame capture
2. **DeGirum Inference Engine**: Manages AI model loading and inference
3. **Face Detection Pipeline**: DeGirum-based detection pipeline
4. **Face Recognition Pipeline**: DeGirum-based recognition pipeline
5. **Face Database**: SQLite-based storage for face embeddings
6. **Display Manager**: Real-time visualization of results
7. **Configuration Manager**: YAML-based configuration system
8. **Stream Reconnection Handler**: Automatic RTSP reconnection logic

## Performance

- **Face Detection**: ~20-30 FPS per camera on DeGirum accelerator
- **Face Recognition**: ~10-15 FPS per camera with database lookup
- **Memory Usage**: ~1.5GB RAM typical usage
- **Latency**: <150ms end-to-end processing time (including network)
- **Network**: Supports multiple RTSP stream formats (H.264, H.265)

## Supported RTSP Camera Formats

- **Video Codecs**: H.264, H.265/HEVC, MJPEG
- **Resolutions**: 720p, 1080p, 4K (auto-scaling)
- **Protocols**: RTSP/RTP, RTSP/TCP, RTSP/UDP
- **Authentication**: Basic, Digest
- **Popular Brands**: Hikvision, Dahua, Axis, Bosch, Uniview, Reolink

## RTSP URL Examples

```yaml
# Hikvision
rtsp_url: "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101"

# Dahua
rtsp_url: "rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0"

# Axis
rtsp_url: "rtsp://user:password@192.168.1.102/axis-media/media.amp"

# Generic IP Camera
rtsp_url: "rtsp://admin:password@192.168.1.103:554/stream1"
```

## Applications

- **Security Systems**: Multi-point access control with IP cameras
- **Retail Analytics**: Customer recognition across multiple camera views
- **Smart Building**: Distributed facial recognition system
- **Perimeter Security**: Multi-camera surveillance with face identification
- **Access Control**: Building entry/exit monitoring

## Troubleshooting

### Common RTSP Issues

1. **Connection Timeout**: Check network connectivity and camera accessibility
2. **Authentication Failed**: Verify username/password and camera settings
3. **Stream Not Found**: Confirm RTSP URL format and stream availability
4. **Codec Issues**: Ensure FFmpeg/GStreamer codec support

### Performance Optimization

1. **Stream Quality**: Adjust camera bitrate and resolution
2. **Network Latency**: Use wired connections when possible
3. **Buffer Management**: Tune buffer sizes for your network conditions
4. **Model Selection**: Choose appropriate DeGirum models for your hardware

Common issues and solutions are documented in the [troubleshooting guide](docs/troubleshooting.md).

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DeGirum for the AI acceleration platform and model zoo
- Raspberry Pi Foundation for the hardware platform
- OpenCV and FFmpeg communities for video processing tools
- IP camera manufacturers for RTSP protocol support