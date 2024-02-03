# Dance Pose Synchronization Analysis

## Overview
The Dance Pose Synchronization Tool offers an innovative approach to compare a user's dance video against a reference, focusing on the precision of pose and movement through pose estimation technologies. Utilizing OpenCV, MediaPipe, and custom synchronization algorithms, this tool calculates the Mean Square Error (MSE) of key joint angles, providing insights into performance alignment and areas for improvement.

## Features
- Pose estimation utilizing MediaPipe.
- Video processing capabilities including mirroring and speed adjustments using FFmpeg.
- Audio synchronization analysis for precise comparison timing.
- MSE-based synchronization scoring for quantitative analysis of pose alignment.
- Customizable video processing options for tailored analysis.

## Prerequisites
Before you start, ensure you have the following installed:
- Python 3.8+
- FFmpeg (for video processing tasks. Ensure ffmpeg is accessible from system's PATH.)
- Praat (for audio analysis and synchronization)


## Installation

### Python Dependencies
Install the required Python libraries with pip:
```sh
pip install numpy opencv-python mediapipe
```

## Usage
Run the tool from CLI. You can also adjust video processing options like mirroring and speed.
```sh
python main.py [options] <reference_video_path> <comparison_video_path>
```
--mirror-reference: Mirror the reference video horizontally.
--mirror-comparison: Mirror the comparison video horizontally.
[speed_factor]: Adjust the speed of both videos by a specific factor. For example, 1.2 to increase speed by 20%.

Example
```sh
python main.py --mirror-reference --mirror-comparison 1.0 reference_video.mp4 comparison_video.mp4
```
