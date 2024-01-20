import unittest
from Video import Video

class TestVideo(unittest.TestCase):
    def setUp(self):
        # Setup with a known video file
        self.testVideoPath = 'path/to/sample_video.mp4'  # Change to an actual video file path
        self.knownFPS = 30  # Change to the known FPS of your video
        self.video = Video(self.testVideoPath)

    def test_getFPS(self):
        # Test if the FPS getter returns the correct value
        self.assertEqual(self.video.getFPS(), self.knownFPS)

    def test_getPath(self):
        # Test if the path getter returns the correct value
        self.assertEqual(self.video.getPath(), self.testVideoPath)

    def test_extractFrames(self):
        # Test if frames are extracted correctly
        frames = self.video.extractFrames()
        # You might want to check the number of frames matches expected based on video length and FPS
        # This is a simple check assuming the video is 1 second and FPS is known
        expectedFrameCount = self.knownFPS * self.video.length  # assuming length is in seconds
        self.assertEqual(len(frames), expectedFrameCount)

if __name__ == '__main__':
    unittest.main()
