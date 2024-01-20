class Video:
    # Assuming some basic structure for Video class, you will define it more appropriately.
    def __init__(self, filePath):
        self.filePath = filePath
        # ... include other necessary attributes and methods

class VideoManager:
    def __init__(self):
        self.referenceVideo = None
        self.studentVideo = None

    def uploadReferenceVideo(self, video):
        """Uploads a reference video.
        
        Args:
            video (Video): The video to be uploaded as reference.
        """
        if isinstance(video, Video):
            self.referenceVideo = video
        else:
            raise ValueError("Uploaded reference is not a video.")

    def uploadStudentVideo(self, video):
        """Uploads a student's video.
        
        Args:
            video (Video): The video to be uploaded as student's video.
        """
        if isinstance(video, Video):
            self.studentVideo = video
        else:
            raise ValueError("Uploaded student video is not a video.")

    def validateVideoCount(self):
        """Validates that both reference and student videos are uploaded.
        
        Returns:
            bool: True if both videos are uploaded, False otherwise.
        """
        return self.referenceVideo is not None and self.studentVideo is not None

    def standardizeFrameRates(self, targetFrameRate):
        """Standardizes the frame rate of both videos to the target frame rate.
        
        Args:
            targetFrameRate (int): The target frame rate to standardize videos.
        """
        if not self.validateVideoCount():
            raise Exception("Cannot standardize frame rates as one or both videos are missing.")

        # Assuming Video class has a method to adjust frame rate
        self.referenceVideo.adjustFrameRate(targetFrameRate)
        self.studentVideo.adjustFrameRate(targetFrameRate)
        # Note: The adjustFrameRate method needs to be defined in the Video class.
