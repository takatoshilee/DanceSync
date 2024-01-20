class FrequencyData:
    def __init__(self, frequencies):
        """Initializes the FrequencyData with an array of frequencies.
        
        Args:
            frequencies (Array[float]): An array of frequency components.
        """
        self.frequencies = frequencies

    def getFrequencies(self):
        """Returns the array of frequency components.
        
        Returns:
            Array[float]: The frequency components.
        """
        return self.frequencies
