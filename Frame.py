class Frame:
    def __init__(self, imageData, timestamp, errorScore=0.0):
        self.errorScore = errorScore  # Initialize error score, defaulting to 0.0
        self.timestamp = timestamp  # Timestamp of the frame in the video
        self.imageData = imageData  # The image data of the frame

    def getErrorScore(self):
        """Returns the error score of the frame."""
        return self.errorScore

    def getTimestamp(self):
        """Returns the timestamp of the frame."""
        return self.timestamp

    def getImageData(self):
        """Returns the image data of the frame."""
        return self.imageData
