# Cat Detector

This is a Python script that detects motion and checks if a cat is present in the video stream. If a cat is detected, the script saves an image of the cat and sends a notification to the user using the ntfy service.

## Requirements

- Python 3.x
- PyTorch
- TorchVision
- Pillow
- OpenCV
- Emoji
- Requests

## Installation

1. Clone the repository:
'pip install -r requirements.txt'

2. Install the required packages:
'pip install -r requirements.txt'

3. Run the script:
## Usage
'python cat_detector.py --rtsp_url rtsp://username:password@ip_address:port/path'

The script takes one command-line argument:

- `--rtsp_url`: The RTSP URL for the camera stream.
- '--topic' : Ntfy topic

The script will write log messages to a file named `cat_detector.log` in the current directory. If a cat is detected, the script will save an image of the cat to the `cats` directory and send a notification to the user using the ntfy service.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.