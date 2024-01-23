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

def run_ffmpeg_command(command):
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command, shell=True, stdout=devnull, stderr=devnull)


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
import glob  
import cv2
from cv2 import VideoWriter_fourcc, VideoWriter
import math
import mediapipe as mp
from statistics import mean
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")


OUTPUT_FOLDER = "output"
OVERWRITE_FLAG = "-y"  # skip if file exists, change to -y for overwrite
PRAAT_EXECUTABLE = "/Applications/Praat.app/Contents/MacOS/Praat"
TIME_INTERVAL = 30  # duration in seconds

# --------------------------------------------------------- VIDEO PROCESSING --------------------------------------------------------------------------------------

def get_clipname(file_path):
    return file_path.split('/')[-1].split(".")[0]

def mirror_video(video_file):
    mirrored_file = f"{OUTPUT_FOLDER}/{get_clipname(video_file) + '_mirrored.mov'}"
    run_ffmpeg_command(f"ffmpeg {OVERWRITE_FLAG} -i {video_file} -vf 'hflip' {mirrored_file}")
    return mirrored_file

def extract_audio_from_video(video_file):
    audio_file = f"{OUTPUT_FOLDER}/{get_clipname(video_file)}_audio.aac"
    run_ffmpeg_command(f"ffmpeg {OVERWRITE_FLAG} -i {video_file} -vn -acodec copy {audio_file}")
    return audio_file

def merge_audio_with_video(video_file, audio_file, output_file):
    video_length = get_video_length(video_file)
    trimmed_audio_file = f"{OUTPUT_FOLDER}/{get_clipname(audio_file)}_trimmed.aac"
    run_ffmpeg_command(f"ffmpeg {OVERWRITE_FLAG} -i {audio_file} -ss 0 -t {video_length} {trimmed_audio_file}")
    run_ffmpeg_command(f"ffmpeg {OVERWRITE_FLAG} -i {video_file} -i {trimmed_audio_file} -c:v copy -c:a aac -strict experimental {output_file}")
    os.remove(trimmed_audio_file)



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

    video_stream, total_frames = get_total_frames(video_file)

    for i in range(total_frames):
        frame_read_success, frame = video_stream.read()  # Read each frame
        if not frame_read_success:
            continue  # Skip to the next frame if current frame is not read successfully

        video_frames.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

        # Detecting landmarks in frame
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

    for i in range(total_frames): 
        frame_read_success, frame = video_stream.read() # Read each frame
        video_frames.append(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB

        frame_landmarks = body_pose.process(frame_rgb)
        pose_landmarks.append(frame_landmarks)

        xy_joint_coords.append([(landmark.x, landmark.y) for landmark in frame_landmarks.pose_landmarks.landmark])

    print(f"Extracted {len(xy_joint_coords)} sets of landmarks from {video_file}")

    return xy_joint_coords, video_frames, pose_landmarks
    


def compare_joint_positions(xy_dancer1, xy_dancer2, frames_dancer1, frames_dancer2, landmarks_dancer1, landmarks_dancer2, frame_rate):
    #DEBUG
    print("Length of xy_dancer1:", len(xy_dancer1))
    print("Length of xy_dancer2:", len(xy_dancer2))

    joint_pairs = [(16, 14), (14, 12), (12, 11), (11, 13), (13, 15), (12, 24), (11, 23), (24, 23), (24, 26), (23, 25), (26, 28), (25, 27)]
    
    joint_names = ["Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", 
                   "Left Elbow", "Left Wrist", "Upper Body", "Right Hip", 
                   "Right Knee", "Left Hip", "Left Knee", "Lower Body"]

    joint_accuracy = {joint: 0 for joint in joint_names}
    joint_count = {joint: 0 for joint in joint_names}

    unsynced_frames = 0
    synchronization_score = 100

    frame_comparison_count = min(len(xy_dancer1), len(xy_dancer2))

    print("Analyzing synchronization...")
    output_video = VideoWriter(f'{OUTPUT_FOLDER}/output.mp4', VideoWriter_fourcc(*'mp4v'), 24.0, (2*720, 1280), isColor=True)

    frame_accuracies = []

    second_accuracies = {}

    for frame_index in range(frame_comparison_count):
        joint_angle_differences = []
        
        joints_dancer1, joints_dancer2 = xy_dancer1[frame_index], xy_dancer2[frame_index]

        if not joints_dancer1 or not joints_dancer2:
            continue
        


        for pair_index, pair in enumerate(joint_pairs):
            joint1, joint2 = pair

            if joint1 >= len(joints_dancer1) or joint2 >= len(joints_dancer1) or \
            joint1 >= len(joints_dancer2) or joint2 >= len(joints_dancer2):
                    continue

            if (joints_dancer1[joint1][0] == joints_dancer1[joint2][0]) or \
                (joints_dancer2[joint1][0] == joints_dancer2[joint2][0]):
                    continue

            gradient_dancer1 = (joints_dancer1[joint1][1] - joints_dancer1[joint2][1]) / (joints_dancer1[joint1][0] - joints_dancer1[joint2][0])
            gradient_dancer2 = (joints_dancer2[joint1][1] - joints_dancer2[joint2][1]) / (joints_dancer2[joint1][0] - joints_dancer2[joint2][0])

            difference = abs((gradient_dancer1 - gradient_dancer2) / gradient_dancer1)
            joint_angle_differences.append(abs(difference))

            if difference < 2:  # example threshold, can be adjusted
                joint_accuracy[joint_names[pair_index]] += 1
            joint_count[joint_names[pair_index]] += 1
        
        if joint_angle_differences:
            # Calculate the mean difference for the frame
            frame_difference = mean(joint_angle_differences)

            frame_accuracy = 100 - frame_difference * 100  # Convert to percentage
            frame_accuracies.append((frame_index, frame_accuracy))

            frame_height, frame_width, _ = frames_dancer1[frame_index].shape
            mpDrawing.draw_landmarks(frames_dancer1[frame_index], landmarks_dancer1[frame_index].pose_landmarks, mpBodyPose.POSE_CONNECTIONS)
            mpDrawing.draw_landmarks(frames_dancer2[frame_index], landmarks_dancer2[frame_index].pose_landmarks, mpBodyPose.POSE_CONNECTIONS)
            comparison_display = np.concatenate((frames_dancer1[frame_index], frames_dancer2[frame_index]), axis=1)

            color = (0, 0, 255) if frame_difference > 10 else (255, 0, 0)
            cv2.putText(comparison_display, f"Diff: {frame_difference:.2f}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            if frame_difference > 10:
                unsynced_frames += 1

            synchronization_score = ((frame_index + 1 - unsynced_frames) / (frame_index + 1)) * 100.0
            cv2.putText(comparison_display, f"Score: {synchronization_score:.2f}%", (frame_width + 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.imshow(str(frame_index), comparison_display)
            output_video.write(comparison_display)
            cv2.waitKey(1)
        else:
            # Optionally handle frames with no valid joint comparisons
            frame_accuracies.append((frame_index, 0)) # Assigning 0% accuracy as an example
    output_video.release()
    
    for i in range(0, len(frame_accuracies), int(frame_rate)):
        second = i // int(frame_rate)
        accuracies_this_second = [acc[1] for acc in frame_accuracies[i:i + int(frame_rate)] if acc]
        if accuracies_this_second:
            average_accuracy = sum(accuracies_this_second) / len(accuracies_this_second)
            second_accuracies[second] = average_accuracy

    if not second_accuracies:
        print("No valid second-level accuracies recorded.")
        return None  # Or some default value, e.g., 0

    min_accuracy_second = min(second_accuracies, key=second_accuracies.get)
    print(f"Lowest accuracy at second {min_accuracy_second} with {second_accuracies[min_accuracy_second]:.2f}% accuracy.")


    print("Accuracy Summary")
    for joint, acc_count in joint_accuracy.items():
        if joint_count[joint] > 0:  # Avoid division by zero
            accuracy_percentage = (acc_count / joint_count[joint]) * 100
            print(f"{joint} Accuracy: {accuracy_percentage:.2f}%")

    min_accuracy_frame = min(frame_accuracies, key=lambda x: x[1])
    print(f"Lowest accuracy at frame {min_accuracy_frame[0]} with {min_accuracy_frame[1]:.2f}% accuracy.")

    return synchronization_score

# --------------------------------------------------------- SYNCING -----------------------------------------------------------------------------------

def convert_video_to_wav(video_file):
    audio_file = f"{OUTPUT_FOLDER}/{get_clipname(video_file)}.wav"
    with suppress_output():
        os.system(f"ffmpeg {OVERWRITE_FLAG} -loglevel panic -hide_banner -i {video_file} {audio_file}")
    return audio_file


def standardize_framerate(video_file):
    standardized_clip = f"{OUTPUT_FOLDER}/{get_clipname(video_file) + '_24fps'}.mov"
    with suppress_output():
        os.system(f"ffmpeg {OVERWRITE_FLAG} -loglevel panic -i {video_file} -filter:v fps=24 {standardized_clip}")
    return standardized_clip



def check_reference_video_length(reference_video, compared_video):
    "Ensure the reference video is longer than the video it's compared to"
    _, reference_video_frames = get_total_frames(reference_video)
    _, compared_video_frames = get_total_frames(compared_video)
    
    print(f"Reference video frames: {reference_video_frames}, Compared video frames: {compared_video_frames}")

    if reference_video_frames <= compared_video_frames:
        print(f"Reference video {reference_video} must be longer than compared video {compared_video}")
        print("Exiting program due to error")
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
    audio_offset = subprocess.check_output(offset_command, shell=True)
    return abs(float(str(audio_offset)[2:-3]))

# --------------------------------------------------------- COMPUTE SYNC ------------------------------------------------------------------------

def equalize_clip_lengths(reference_clip, compared_clip, time_offset):
    compared_clip_duration = get_video_length(compared_clip)
    trimmed_ref_clip = f"{OUTPUT_FOLDER}/{get_clipname(reference_clip) + '_trimmed.mov'}"
    trimmed_compared_clip = f"{OUTPUT_FOLDER}/{get_clipname(compared_clip) + '_trimmed.mov'}"
    with suppress_output():
        os.system(f"ffmpeg {OVERWRITE_FLAG} -i {reference_clip} -ss {time_offset} -t {compared_clip_duration} {trimmed_ref_clip}")
        os.system(f"ffmpeg {OVERWRITE_FLAG} -i {compared_clip} -ss 0 -t {compared_clip_duration} {trimmed_compared_clip}")
    return trimmed_ref_clip, trimmed_compared_clip


def delete_temporary_videos():
    """
    Removes all temporary video files created during the processing (both trimmed and framerate-adjusted videos).
    """
    trimmed_files = glob.glob(f"{OUTPUT_FOLDER}/*trimmed.mov")
    framerate_adjusted_files = glob.glob(f"{OUTPUT_FOLDER}/*24fps.mov")

    for file in trimmed_files + framerate_adjusted_files:
        os.remove(file)


# --------------------------------------------------------- PREPARE VIDEOS --------------------------------------------------------------------------------------

def change_video_speed(video_file, speed_factor):
    output_file = f"{OUTPUT_FOLDER}/{get_clipname(video_file)}_speed_{speed_factor}.mov"
    os.system(f"ffmpeg {OVERWRITE_FLAG} -i {video_file} -filter:v 'setpts={1/speed_factor}*PTS' -filter:a 'atempo={speed_factor}' {output_file}")
    return output_file

def calculate_frame_rate(video_file):
    video_capture = cv2.VideoCapture(video_file)
    print(f"Attempting to open video file {video_file}")

    if not video_capture.isOpened():
        print(f"Failed to open video file {video_file} for frame rate calculation.")
        return None
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()
    return frame_rate



import numpy as np

def calculate_angle(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    angle = np.arctan2(a[1], a[0]) - np.arctan2(b[1], b[0])
    return np.abs(angle)

def get_joint_position(joint, xy_coordinates):
    if type(joint) == tuple:
        if len(joint) == 2 and joint[1] == 'midpoint':
            # Midpoint calculation
            j1, j2 = joint[0]
            if j1 < len(xy_coordinates) and j2 < len(xy_coordinates) and xy_coordinates[j1] and xy_coordinates[j2]:
                # Added check for tuple values
                if isinstance(xy_coordinates[j1], tuple) and isinstance(xy_coordinates[j2], tuple):
                    return [(xy_coordinates[j1][0] + xy_coordinates[j2][0]) / 2,
                            (xy_coordinates[j1][1] + xy_coordinates[j2][1]) / 2]
                else:
                    return None  # Or handle error
            else:
                return None
        else:
            # Regular tuple with two joints
            j1, j2 = joint
            if j1 < len(xy_coordinates) and xy_coordinates[j1]:
                return xy_coordinates[j1]
            if j2 < len(xy_coordinates) and xy_coordinates[j2]:
                return xy_coordinates[j2]
            return None
    else:
        # Single joint index
        if joint < len(xy_coordinates) and xy_coordinates[joint]:
            return xy_coordinates[joint]
        else:
            return None



def calculate_mse(xy_dancer1, xy_dancer2, joint_sets, penalty_per_missing_joint=0.25 * (180 ** 2)):
    print(f"Calculating MSE for {len(joint_sets)} joint sets")

    angle_differences = []
    missing_joint_count = 0

    for joint1, joint2, joint3 in joint_sets:
        # Get positions for each joint
        p1_dancer1 = get_joint_position(joint1, xy_dancer1)
        p2_dancer1 = get_joint_position(joint2, xy_dancer1)
        p3_dancer1 = get_joint_position(joint3, xy_dancer1)

        p1_dancer2 = get_joint_position(joint1, xy_dancer2)
        p2_dancer2 = get_joint_position(joint2, xy_dancer2)
        p3_dancer2 = get_joint_position(joint3, xy_dancer2)

        # Check if any joint positions are None
        if None in (p1_dancer1, p2_dancer1, p3_dancer1, p1_dancer2, p2_dancer2, p3_dancer2):
            missing_joint_count += 1
            print(f"Missing joint data for frame {joint1}, {joint2}, {joint3}")
            continue

        # Calculate the angles and differences
        angle1 = calculate_angle(p1_dancer1, p2_dancer1, p3_dancer1)
        angle2 = calculate_angle(p1_dancer2, p2_dancer2, p3_dancer2)
        angle_differences.append((angle1 - angle2) ** 2)

    # Calculate MSE and ensure it's a scalar
    mse = sum(angle_differences) / len(angle_differences) if angle_differences else 0
    mse += missing_joint_count * penalty_per_missing_joint
    return mse.item() if isinstance(mse, np.ndarray) and mse.size == 1 else mse


def calculate_synchronization_score(mse, alpha=0.001):
    score = 1 - np.tanh(alpha * mse)
    # Convert score to a scalar if it's a NumPy array
    if isinstance(score, np.ndarray):
        if score.size == 1:
            return score.item()
        else:
            return score.mean()  # or any other appropriate aggregation
    return score





# Angular Calculation: Joint Sets
joint_sets_angular = [
    (12, 14, 16), # Right Arm: Shoulder, Elbow, Wrist
    (11, 13, 15), # Left Arm: Shoulder, Elbow, Wrist
    (24, 26, 28), # Right Leg: Hip, Knee, Ankle
    (23, 25, 27), # Left Leg: Hip, Knee, Ankle
    ((12, 24), (24, 23), (23, 11)), # Upper Body: Right Shoulder, Midpoint of Hips, Left Shoulder
    (((11, 12), 'midpoint'), 0, 10)  # Neck and Head: Midpoint of Shoulders, Neck, Head Top
]

# Function for MSE-based comparison using angular calculation
def mse_based_comparison(xy_dancer1, xy_dancer2):
    mse = calculate_mse(xy_dancer1, xy_dancer2, joint_sets_angular)
    synchronization_score_mse = calculate_synchronization_score(mse)
    print(f"Synchronization score (MSE-based): {synchronization_score_mse:.2f}%")



if __name__ == "__main__":
    # Default speed factor
    default_speed_factor = 1.0
    speed_factor = default_speed_factor
    frame_rate = None  # Initialize frame_rate


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
        print("Exiting program due to error")
        sys.exit(-1)

    reference_video = args[0]
    comparison_video = args[1]

    

    print(f"Processing reference video: {reference_video}")
    print(f"Processing comparison video: {comparison_video}")


    # Mirror videos if flags are set
    if mirror_reference:
        reference_video = mirror_video(reference_video)
    if mirror_comparison:
        comparison_video = mirror_video(comparison_video)

    # Change speed of videos
    reference_video = change_video_speed(reference_video, speed_factor)
    comparison_video = change_video_speed(comparison_video, speed_factor)

    # Extract landmarks from both videos
    xy_ref_dancer, ref_dancer_frames, ref_dancer_landmarks = extract_landmarks(reference_video)
    xy_comp_dancer, comp_dancer_frames, comp_dancer_landmarks = extract_landmarks(comparison_video)

    # Check if landmarks were successfully extracted
    if not xy_ref_dancer or not xy_comp_dancer:
        print("Failed to extract landmarks from one or both videos.")
        sys.exit(-1)


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
    
    # Frame rate calculation using trimmed video
    frame_rate = calculate_frame_rate(trimmed_reference_video)
    if frame_rate is None or frame_rate == 0:
        print("Error calculating frame rate. Exiting.")
        sys.exit(-1)

    print("Calculated Frame Rate:", frame_rate)
    
    # Analyzing dance synchronization
    print(f"Reference model: {trimmed_reference_video}, Comparison: {trimmed_comparison_video} \n")
    xy_ref_dancer, ref_dancer_frames, ref_dancer_landmarks = extract_landmarks(trimmed_reference_video)
    print(f"Extracting landmarks from {trimmed_reference_video}")
    # Call to extract_landmarks
    print("Landmarks extraction complete")

    xy_comp_dancer, comp_dancer_frames, comp_dancer_landmarks = extract_landmarks(trimmed_comparison_video)

    # Gradient-based comparison
    if frame_rate is not None:
        synchronization_score_gradient = compare_joint_positions(xy_ref_dancer, xy_comp_dancer, ref_dancer_frames, comp_dancer_frames, ref_dancer_landmarks, comp_dancer_landmarks, frame_rate)
        print(f"\nYour synchronization with the reference model (Gradient-based) is {synchronization_score_gradient:.2f}%.")

    # MSE-based comparison
    mse_based_comparison(xy_ref_dancer, xy_comp_dancer)

    # Cleaning up temporary files
    # delete_temporary_videos()

    # After creating output.mp4
    extracted_audio = extract_audio_from_video(reference_video)  # Extract audio from the reference video
    final_output_video = f"{OUTPUT_FOLDER}/final_output_with_audio.mp4"
    merge_audio_with_video(f"{OUTPUT_FOLDER}/output.mp4", extracted_audio, final_output_video)

    # Cleaning up temporary files
    delete_temporary_videos()
# --------------------------------------------------------- MAIN --------------------------------------------------------------------------------------
else:
    # Assigning already trimmed videos for comparison
    trimmed_reference_video = sys.argv[1]
    trimmed_comparison_video = sys.argv[2]



# Analyzing dance synchronization
print(f"Reference model: {trimmed_reference_video}, Comparison: {trimmed_comparison_video} \n")
xy_ref_dancer, ref_dancer_frames, ref_dancer_landmarks = extract_landmarks(trimmed_reference_video)
xy_comp_dancer, comp_dancer_frames, comp_dancer_landmarks = extract_landmarks(trimmed_comparison_video)

# Check if frame_rate is available before comparing joint positions
if frame_rate is not None:
    synchronization_score = compare_joint_positions(xy_ref_dancer, xy_comp_dancer, ref_dancer_frames, comp_dancer_frames, ref_dancer_landmarks, comp_dancer_landmarks, frame_rate)
    print(f"\n Your synchronization with the reference model is {synchronization_score:.2f}%.")
else:
    print("Frame rate is not available. Cannot proceed with synchronization score calculation.")


# Calculate synchronization score
synchronization_score = calculate_synchronization_score(mse)
print(f"Synchronization score: {synchronization_score:.2f}%")

# New MSE-based comparison
mse_based_comparison(xy_ref_dancer, xy_comp_dancer)

# After creating output.mp4
extracted_audio = extract_audio_from_video(reference_video)  # Extract audio from the reference video
final_output_video = f"{OUTPUT_FOLDER}/final_output_with_audio.mp4"
merge_audio_with_video(f"{OUTPUT_FOLDER}/output.mp4", extracted_audio, final_output_video)

# Cleaning up temporary files
delete_temporary_videos()