import unittest
from Frame import Frame

class TestFrame(unittest.TestCase):
    def setUp(self):
        # Setup a frame with known values to test against
        self.testImageData = b'test_image_data'  # Simulated image data
        self.testTimestamp = 10.5  # Simulated timestamp
        self.testErrorScore = 0.1  # Simulated error score
        self.frame = Frame(self.testImageData, self.testTimestamp, self.testErrorScore)

    def test_getErrorScore(self):
        # Test if the error score getter returns the correct value
        self.assertEqual(self.frame.getErrorScore(), self.testErrorScore)

    def test_getTimestamp(self):
        # Test if the timestamp getter returns the correct value
        self.assertEqual(self.frame.getTimestamp(), self.testTimestamp)

    def test_getImageData(self):
        # Test if the image data getter returns the correct value
        self.assertEqual(self.frame.getImageData(), self.testImageData)

if __name__ == '__main__':
    unittest.main()
