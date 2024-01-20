from vid_utils import Video, concatenate_videos

videos = [
  Video(speed=1.0, path="C:/temp/my_video_1.mp4"),
  Video(speed=2.0, path="C:/temp/my_video_2.mp4"),
  Video(speed=0.5, path="C:/temp/my_video_3.mp4"),
]

concatenate_videos(videos=videos, output_file=f"C:/temp/output_video.mp4")
