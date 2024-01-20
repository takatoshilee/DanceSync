
import os

"""
TODO 
add mirror function done

add speed funciton done -> MAKE INTO terminal args

OOP

GUI

# Cleaning up temporary files
# delete_temporary_videos()


real time testing -> use posenet instead of mediapipe
"""


#SUPPRESSION
from contextlib import contextmanager
from contextlib import suppress


# Suppressing warnings
import warnings
warnings.filterwarnings("ignore")

# suppress mediapipe logs
os.environ['GLOG_minloglevel'] = '2' 

@contextmanager
def suppress_output():
    new_stdout, new_stderr = open(os.devnull, 'w'), open(os.devnull, 'w')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = new_stdout, new_stderr
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        new_stdout.close()
        new_stderr.close()


#OPTIONAL END
            

import subprocess
from glob import glob
import cv2
from cv2 import VideoWriter_fourcc, VideoWriter
import math
import mediapipe as mp
from statistics import mean
import numpy as np
import sys

# Suppressing warnings
import warnings
warnings.filterwarnings("ignore")


OUTPUT_FOLDER = "output"
OVERWRITE_FLAG = "-y"  # skip if file exists, change to -y for overwrite
PRAAT_EXECUTABLE = "/Applications/Praat.app/Contents/MacOS/Praat"
TIME_INTERVAL = 30  # duration in seconds

# --------------------------------------------------------- VIDEO PROCESSING --------------------------------------------------------------------------------------

# Video Processing Functions
def get_clipname(file_path):
    return file_path.split('/')[-1].split(".")[0]

def mirror_video(video_file):
    mirrored_file = f"{OUTPUT_FOLDER}/{get_clipname(video_file) + '_mirrored.mov'}"
    os.system(f"ffmpeg {OVERWRITE_FLAG} -i {video_file} -vf 'hflip' {mirrored_file}")
    return mirrored_file

# Function to process command line arguments and extract mirrored videos
"""
def parse_and_mirror_videos(args):
    mirrored_videos = []
    videos_to_process = []
    mirror_next = False

    for arg in args:
        if arg == '--mirror':
            mirror_next = True
        elif mirror_next:
            mirrored_video = mirror_video(arg)
            mirrored_videos.append(mirrored_video)
            videos_to_process.append(mirrored_video)
            mirror_next = False
        else:
            videos_to_process.append(arg)

    return videos_to_process, mirrored_videos
    mirrored_videos = []
    videos_to_process = []
    mirror_next = False

    # Define the index of the current video being processed
    current_video_index = 0

    for arg in args:
        if arg == '--mirror':
            mirror_next = True
            continue

        if mirror_next:
            mirrored_video = mirror_video(arg)
            mirrored_videos.append(mirrored_video)
            videos_to_process.append(mirrored_video)
            mirror_next = False
        else:
            if current_video_index < len(videos_to_process):
                videos_to_process[current_video_index] = arg
            else:
                videos_to_process.append(arg)
            current_video_index += 1

    return videos_to_process, mirrored_videos
    mirrored_videos = []
    videos_to_process = []

    for i, arg in enumerate(args):
        if arg == '--mirror':
            # Mirror the next video
            mirrored_video = mirror_video(args[i + 1])
            mirrored_videos.append(mirrored_video)
            videos_to_process.append(mirrored_video)
        elif i == 0 or (i > 0 and args[i - 1] != '--mirror'):
            # Add non-mirrored videos
            videos_to_process.append(arg)

    return videos_to_process, mirrored_videos


# Main execution block
if __name__ == "__main__":
    # Check if enough arguments are passed
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <reference_video> <comparison_video> [--mirror-reference] [--mirror-comparison]")
        sys.exit(-1)

    # Initialize flags
    mirror_reference = '--mirror-reference' in sys.argv
    mirror_comparison = '--mirror-comparison' in sys.argv

    # Remove flags from arguments
    args = [arg for arg in sys.argv[1:] if arg not in ['--mirror-reference', '--mirror-comparison']]

    # Assigning command line arguments to variables
    reference_video = args[0]
    comparison_video = args[1]

    # Mirror videos if flags are set
    if mirror_reference:
        reference_video = mirror_video(reference_video)
    if mirror_comparison:
        comparison_video = mirror_video(comparison_video)


    # Creating the output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Verifying command line arguments
    if len(sys.argv) < 3:
        print(f"Correct Usage:\n {sys.argv[0]} <reference_video> <comparison_video>")
        sys.exit(-1)

    # Check if '--mirror' flag is used and mirror the specified video
    if '--mirror' in sys.argv:
        mirror_index = sys.argv.index('--mirror')
        video_to_mirror = sys.argv[mirror_index + 1]

        # Mirror the specified video
        mirrored_video = mirror_video(video_to_mirror)

        # Replace the original video with the mirrored version in the argument list
        sys.argv[mirror_index + 1] = mirrored_video

    # Assigning command line arguments to variables
    reference_video = sys.argv[1]
    comparison_video = sys.argv[2]
"""

def get_video_length(file_path):
    video_capture = cv2.VideoCapture(file_path)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_length = (total_frames / frame_rate)
    return video_length

def get_total_frames(video_file):
    video_capture = cv2.VideoCapture(video_file)
    total_frame_count = int(math.floor(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    return video_capture, total_frame_count

# utilities
mpDrawing = mp.solutions.drawing_utils
mpBodyPose = mp.solutions.pose

def extract_landmarks(video_file):
    body_pose = mpBodyPose.Pose()  # Initialize the pose detection

    xy_joint_coords = []  # Focusing on x and y coordinates only
    video_frames = []
    pose_landmarks = []

    # Video capture and frame count
    video_stream, total_frames = get_total_frames(video_file)

    # Processing the video
    for i in range(total_frames):
        frame_read_success, frame = video_stream.read()  # Read each frame
        if not frame_read_success:
            continue  # Skip to the next frame if current frame is not read successfully

        video_frames.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

        # Detecting landmarks in the frame
        frame_landmarks = body_pose.process(frame_rgb)

        if frame_landmarks.pose_landmarks:
            xy_joint_coords.append([(landmark.x, landmark.y) for landmark in frame_landmarks.pose_landmarks.landmark])
            pose_landmarks.append(frame_landmarks)
        else:
            xy_joint_coords.append([])
            pose_landmarks.append(None)

    return xy_joint_coords, video_frames, pose_landmarks
    body_pose = mpBodyPose.Pose() # Initialize the pose detection

    xy_joint_coords = [] # Focusing on x and y coordinates only
    video_frames = []
    pose_landmarks = []

    # Video capture and frame count
    video_stream, total_frames = get_total_frames(video_file)

    # Processing the video
    for i in range(total_frames): 
        frame_read_success, frame = video_stream.read() # Read each frame
        video_frames.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB

        # Detecting landmarks in the frame
        frame_landmarks = body_pose.process(frame_rgb)
        pose_landmarks.append(frame_landmarks)

        # Extracting x and y coordinates of joint positions
        xy_joint_coords.append([(landmark.x, landmark.y) for landmark in frame_landmarks.pose_landmarks.landmark])

    return xy_joint_coords, video_frames, pose_landmarks

def compare_joint_positions(xy_dancer1, xy_dancer2, frames_dancer1, frames_dancer2, landmarks_dancer1, landmarks_dancer2):
    #DEBUG
    print("Length of xy_dancer1:", len(xy_dancer1))
    print("Length of xy_dancer2:", len(xy_dancer2))

    # Joint pairs for comparison
    joint_pairs = [(16, 14), (14, 12), (12, 11), (11, 13), (13, 15), (12, 24), (11, 23), (24, 23), (24, 26), (23, 25), (26, 28), (25, 27)]
    
    # Corresponding joint names for each pair
    joint_names = ["Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", 
                   "Left Elbow", "Left Wrist", "Upper Body", "Right Hip", 
                   "Right Knee", "Left Hip", "Left Knee", "Lower Body"]

    # Initialize dictionaries to track accuracy and count for each joint
    joint_accuracy = {joint: 0 for joint in joint_names}
    joint_count = {joint: 0 for joint in joint_names}

    # Tracking the number of frames where dancers are not synchronized
    unsynced_frames = 0
    synchronization_score = 100

    # Determine the number of frames for comparison
    frame_comparison_count = min(len(xy_dancer1), len(xy_dancer2))

    print("Analyzing synchronization...")
    # Creating the output video
    output_video = VideoWriter(f'{OUTPUT_FOLDER}/output.mp4', VideoWriter_fourcc(*'mp4v'), 24.0, (2*720, 1280), isColor=True)


    for frame_index in range(frame_comparison_count):
        # Calculate joint angle differences for each frame
        joint_angle_differences = []

        # Retrieve joint positions for the current frame
        joints_dancer1, joints_dancer2 = xy_dancer1[frame_index], xy_dancer2[frame_index]

        for pair_index, pair in enumerate(joint_pairs):
            joint1, joint2 = pair

            # Calculate gradients (slopes) for each dancer
            gradient_dancer1 = (joints_dancer1[joint1][1] - joints_dancer1[joint2][1]) / (joints_dancer1[joint1][0] - joints_dancer1[joint2][0])
            gradient_dancer2 = (joints_dancer2[joint1][1] - joints_dancer2[joint2][1]) / (joints_dancer2[joint1][0] - joints_dancer2[joint2][0])

            # Compute the difference in gradients
            difference = abs((gradient_dancer1 - gradient_dancer2) / gradient_dancer1)
            joint_angle_differences.append(abs(difference))

            # Update joint accuracy
            if difference < 2:  # example threshold, can be adjusted
                joint_accuracy[joint_names[pair_index]] += 1
            joint_count[joint_names[pair_index]] += 1

        # Calculate the mean difference for the frame
        frame_difference = mean(joint_angle_differences)

        # Create live comparison display
        frame_height, frame_width, _ = frames_dancer1[frame_index].shape
        mpDrawing.draw_landmarks(frames_dancer1[frame_index], landmarks_dancer1[frame_index].pose_landmarks, mpBodyPose.POSE_CONNECTIONS)
        mpDrawing.draw_landmarks(frames_dancer2[frame_index], landmarks_dancer2[frame_index].pose_landmarks, mpBodyPose.POSE_CONNECTIONS)
        comparison_display = np.concatenate((frames_dancer1[frame_index], frames_dancer2[frame_index]), axis=1)

        # Set color based on difference magnitude
        color = (0, 0, 255) if frame_difference > 10 else (255, 0, 0)
        cv2.putText(comparison_display, f"Diff: {frame_difference:.2f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Increment unsynced frames count if needed
        if frame_difference > 10:
            unsynced_frames += 1

        # Update live synchronization score
        synchronization_score = ((frame_index + 1 - unsynced_frames) / (frame_index + 1)) * 100.0
        cv2.putText(comparison_display, f"Score: {synchronization_score:.2f}%", (frame_width + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow(str(frame_index), comparison_display)
        output_video.write(comparison_display)
        cv2.waitKey(1)

    output_video.release()

    # Print Accuracy Summary
    print("Accuracy Summary")
    for joint, acc_count in joint_accuracy.items():
        if joint_count[joint] > 0:  # Avoid division by zero
            accuracy_percentage = (acc_count / joint_count[joint]) * 100
            print(f"{joint} Accuracy: {accuracy_percentage:.2f}%")

    return synchronization_score
    # Joint pairs for comparison
    joint_pairs = [(16, 14), (14, 12), (12, 11), (11, 13), (13, 15), (12, 24), (11, 23), (24, 23), (24, 26), (23, 25), (26, 28), (25, 27)]
    
    # Corresponding joint names for each pair
    joint_names = ["Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", 
                   "Left Elbow", "Left Wrist", "Upper Body", "Right Hip", 
                   "Right Knee", "Left Hip", "Left Knee", "Lower Body"]  # Adjusted names

    # Initialize dictionaries to track accuracy and count for each joint
    joint_accuracy = {joint: 0 for joint in joint_names}
    joint_count = {joint: 0 for joint in joint_names}


    # Tracking the number of frames where dancers are not synchronized
    unsynced_frames = 0
    synchronization_score = 100

    # Determine the number of frames for comparison
    frame_comparison_count = min(len(xy_dancer1), len(xy_dancer2))

    print("Analyzing synchronization...")
    # Creating the output video
    output_video = VideoWriter(f'{OUTPUT_FOLDER}/output.mp4', VideoWriter_fourcc(*'mp4v'), 24.0, (2*720, 1280), isColor=True)

    for frame_index in range(frame_comparison_count):
        # Calculate joint angle differences for each frame
        joint_angle_differences = []

        # Retrieve joint positions for the current frame
        joints_dancer1, joints_dancer2 = xy_dancer1[frame_index], xy_dancer2[frame_index]

        for pair_index, pair in enumerate(joint_pairs):
            joint1, joint2 = pair

            # Calculate gradients (slopes) for each dancer
            gradient_dancer1 = (joints_dancer1[joint1][1] - joints_dancer1[joint2][1]) / (joints_dancer1[joint1][0] - joints_dancer1[joint2][0])
            gradient_dancer2 = (joints_dancer2[joint1][1] - joints_dancer2[joint2][1]) / (joints_dancer2[joint1][0] - joints_dancer2[joint2][0])

            # Debugging: Print gradients and differences
            print(f"Frame {frame_index}, Joint Pair {joint_names[pair_index]}: Gradient Dancer 1 = {gradient_dancer1}, Gradient Dancer 2 = {gradient_dancer2}")


            # Compute the difference in gradients
            difference = abs((gradient_dancer1 - gradient_dancer2) / gradient_dancer1)
            joint_angle_differences.append(abs(difference))

            # Debugging: Print difference
            print(f"Frame {frame_index}, Joint Pair {joint_names[pair_index]}: Difference = {difference}")

            # Update joint accuracy
            if difference < 2:  # example threshold, can be adjusted
                joint_accuracy[joint_names[pair_index]] += 1
            joint_count[joint_names[pair_index]] += 1

        # Calculate the mean difference for the frame
        frame_difference = mean(joint_angle_differences)

        # Create live comparison display
        frame_height, frame_width, _ = frames_dancer1[frame_index].shape
        mpDrawing.draw_landmarks(frames_dancer1[frame_index], landmarks_dancer1[frame_index].pose_landmarks, mpBodyPose.POSE_CONNECTIONS)
        mpDrawing.draw_landmarks(frames_dancer2[frame_index], landmarks_dancer2[frame_index].pose_landmarks, mpBodyPose.POSE_CONNECTIONS)
        comparison_display = np.concatenate((frames_dancer1[frame_index], frames_dancer2[frame_index]), axis=1)

        # Set color based on difference magnitude
        color = (0, 0, 255) if frame_difference > 10 else (255, 0, 0)
        cv2.putText(comparison_display, f"Diff: {frame_difference:.2f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Increment unsynced frames count if needed
        if frame_difference > 10:
            unsynced_frames += 1

        # Update live synchronization score
        synchronization_score = ((frame_index + 1 - unsynced_frames) / (frame_index + 1)) * 100.0
        cv2.putText(comparison_display, f"Score: {synchronization_score:.2f}%", (frame_width + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow(str(frame_index), comparison_display)
        output_video.write(comparison_display)
        cv2.waitKey(1)

    output_video.release()

    # Print Accuracy Summary
    print("Accuracy Summary")
    for joint, acc_count in joint_accuracy.items():
        if joint_count[joint] > 0:  # Avoid division by zero
            accuracy_percentage = (acc_count / joint_count[joint]) * 100
            print(f"{joint} Accuracy: {accuracy_percentage:.2f}%")


        return synchronization_score

# --------------------------------------------------------- SYNCING -----------------------------------------------------------------------------------

def convert_video_to_wav(video_file):
    audio_file = f"{OUTPUT_FOLDER}/{get_clipname(video_file)}.wav"
    os.system(f"ffmpeg {OVERWRITE_FLAG} -loglevel panic -hide_banner -i {video_file} {audio_file} > /dev/null 2>&1")
    return audio_file

def standardize_framerate(video_file):
    standardized_clip = f"{OUTPUT_FOLDER}/{get_clipname(video_file) + '_24fps'}.mov"
    with suppress_output():
        os.system(f"ffmpeg {OVERWRITE_FLAG} -loglevel panic -i {video_file} -filter:v fps=24 {standardized_clip} > /dev/null 2>&1")
    return standardized_clip


def check_reference_video_length(reference_video, compared_video):
    "Ensure the reference video is longer than the video it's compared to"
    _, reference_video_frames = get_total_frames(reference_video)
    _, compared_video_frames = get_total_frames(compared_video)
    if reference_video_frames <= compared_video_frames:
        print(f"Reference video {reference_video} must be longer than compared video {compared_video}")
        sys.exit(-1)

def convert_video_to_wav(video_file):
    audio_file = f"{OUTPUT_FOLDER}/{get_clipname(video_file)}.wav"
    with suppress_output():
        os.system(f"ffmpeg {OVERWRITE_FLAG} -loglevel panic -hide_banner -i {video_file} {audio_file} > /dev/null 2>&1")
    return audio_file

def detect_audio_offset(reference_audio, comparison_audio):
    """
    Detects the audio offset between two WAV files. It compares the reference audio with the comparison audio.
    IMPORTANT: Assumes that the input audio files might be of different lengths.
    The offset is used to synchronize clips, ensuring they are compared over the same choreography section.
    """
    initial_offset = 0
    offset_command = f"{PRAAT_EXECUTABLE} --run 'crosscorrelate.praat' {reference_audio} {comparison_audio} {initial_offset} {TIME_INTERVAL}"
    # The actual offset detection logic is handled in a separate Praat script.
    audio_offset = subprocess.check_output(offset_command, shell=True)
    # Formatting to convert the output from byte string to float.
    return abs(float(str(audio_offset)[2:-3]))

# --------------------------------------------------------- COMPUTE SYNC ------------------------------------------------------------------------

def equalize_clip_lengths(reference_clip, compared_clip, time_offset):
    """
    Trims the reference and compared clips to the same length based on the duration of the compared clip.
    """
    compared_clip_duration = get_video_length(compared_clip)

    trimmed_ref_clip = f"{OUTPUT_FOLDER}/{get_clipname(reference_clip) + '_trimmed.mov'}"
    trimmed_compared_clip = f"{OUTPUT_FOLDER}/{get_clipname(compared_clip) + '_trimmed.mov'}"

    trim_ref_command = f"ffmpeg {OVERWRITE_FLAG} -i {reference_clip} -ss {time_offset} -t {compared_clip_duration} {trimmed_ref_clip}"
    os.system(trim_ref_command)
    trim_compared_command = f"ffmpeg {OVERWRITE_FLAG} -i {compared_clip} -ss 0 -t {compared_clip_duration} {trimmed_compared_clip}"
    os.system(trim_compared_command)

    return trimmed_ref_clip, trimmed_compared_clip

def delete_temporary_videos():
    """
    Removes all temporary video files created during the processing (both trimmed and framerate-adjusted videos).
    """
    remove_trimmed_command = "rm *trimmed.mov"
    os.system(remove_trimmed_command)
    remove_framerate_adjusted_command = "rm *24fps.mov"
    os.system(remove_framerate_adjusted_command)


# --------------------------------------------------------- PREPARE VIDEOS --------------------------------------------------------------------------------------

# Setup before execution
# Command to run: python dance.py video/chuu.mov video/cyves.mov
# post processing speed change

def change_video_speed(video_file, speed_factor):
    output_file = f"{OUTPUT_FOLDER}/{get_clipname(video_file)}_speed_{speed_factor}.mov"
    os.system(f"ffmpeg {OVERWRITE_FLAG} -i {video_file} -filter:v 'setpts={1/speed_factor}*PTS' -filter:a 'atempo={speed_factor}' {output_file}")
    return output_file


if __name__ == "__main__":
    # Default speed factor
    default_speed_factor = 0.5
    speed_factor = default_speed_factor

    # Check if the first argument is a speed factor
    try:
        speed_factor = float(sys.argv[1])
        arg_start_index = 2
    except ValueError:
        arg_start_index = 1

    # Initialize flags
    mirror_reference = '--mirror-reference' in sys.argv
    mirror_comparison = '--mirror-comparison' in sys.argv

    # Extract video file paths
    args = [arg for arg in sys.argv[arg_start_index:] if arg not in ['--mirror-reference', '--mirror-comparison']]

    if len(args) < 2:
        print(f"Usage: {sys.argv[0]} [speed_factor] <reference_video> <comparison_video> [--mirror-reference] [--mirror-comparison]")
        sys.exit(-1)

    reference_video = args[0]
    comparison_video = args[1]

    # Mirror videos if flags are set
    if mirror_reference:
        reference_video = mirror_video(reference_video)
    if mirror_comparison:
        comparison_video = mirror_video(comparison_video)

    # Change speed of videos
    reference_video = change_video_speed(reference_video, speed_factor)
    comparison_video = change_video_speed(comparison_video, speed_factor)

# Complete comparison process
if not (len(sys.argv) > 3 and sys.argv[3] == '--compare-only'):
    print(f"Initial videos: {reference_video} {comparison_video}")
    reference_video_24fps, comparison_video_24fps = standardize_framerate(reference_video), standardize_framerate(comparison_video)

    # Checking length of reference video
    print(f'Reference video: {reference_video}, Comparison video: {comparison_video}')
    check_reference_video_length(reference_video, comparison_video)

    # Audio conversion for synchronization analysis
    reference_audio, comparison_audio = convert_video_to_wav(reference_video), convert_video_to_wav(comparison_video)

    # Finding the audio time offset
    audio_offset = detect_audio_offset(reference_audio, comparison_audio)

    # Trimming videos for length synchronization
    trimmed_reference_video, trimmed_comparison_video = equalize_clip_lengths(reference_video_24fps, comparison_video_24fps, audio_offset)
    print(trimmed_reference_video, trimmed_comparison_video)
# --------------------------------------------------------- MAIN --------------------------------------------------------------------------------------
else:
    # Assigning already trimmed videos for comparison
    trimmed_reference_video = sys.argv[1]
    trimmed_comparison_video = sys.argv[2]

# Analyzing dance synchronization
print(f"Reference model: {trimmed_reference_video}, Comparison: {trimmed_comparison_video} \n")
xy_ref_dancer, ref_dancer_frames, ref_dancer_landmarks = extract_landmarks(trimmed_reference_video)
xy_comp_dancer, comp_dancer_frames, comp_dancer_landmarks = extract_landmarks(trimmed_comparison_video)

# Calculating synchronization score
synchronization_score = compare_joint_positions(xy_ref_dancer, xy_comp_dancer, ref_dancer_frames, comp_dancer_frames, ref_dancer_landmarks, comp_dancer_landmarks)
print(f"\n Your synchronization with the reference model is {synchronization_score:.2f}%.")


# Cleaning up temporary files
# delete_temporary_videos()
