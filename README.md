# Traffic Sign Detection with YOLOv8 and OpenVINO

## Overview
This project implements real-time traffic sign detection using the YOLOv8 model from Ultralytics. It processes images or videos in parallel using Python's `multiprocessing` with the `'spawn'` method to ensure CUDA compatibility. The project also conceptually integrates Intel's OpenVINO toolkit for model optimization and deployment on Intel hardware.

Developed for the Intel Campus Ambassador program, this showcases my ability to leverage Intel technologies like OpenVINO for AI inference acceleration.

## Features
- Detects traffic signs in images or videos using YOLOv8.
- Parallel frame processing for videos using multiprocessing.
- Exports the model to ONNX format, ready for OpenVINO optimization.
- Tested in Google Colab with GPU support.

## Prerequisites
- Python 3.8+
- GPU support (optional but recommended)
- Dependencies listed in `requirements.txt`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/TrafficSignDetection_YOLO_OpenVINO.git
   cd TrafficSignDetection_YOLO_OpenVINO
## Testing
1. Unzip TrafficSignDetection_YOLO_OpenVINO.zip locally.
2. Install dependencies: pip install -r requirements.txt
3. Run: python main.py --input sample_image.jpg --output output
4. Check the output/ folder for detected_image.png
## OpenVINO Integration
Convert to IR:
\\bash
mo.py --input_model yolov8n.onnx --output_dir openvino_model
## Inference on Intel Hardware:
\\python
from openvino.inference_engine import IECore
ie = IECore()
net = ie.read_network(model="openvino_model/model.xml", weights="openvino_model/model.bin")
exec_net = ie.load_network(network=net, device_name="CPU")
Benefits: OpenVINO provides 2-3x speedup on Intel CPUs and up to 10x on GPUs via INT8 quantization and hardware acceleration.
Notes
Includes sample_image.jpg (STOP sign) and sample_traffic_video.mp4 for testing.
Replace with your own traffic-related input for custom results.
Multiprocessing uses 'spawn' to avoid CUDA errors.
License