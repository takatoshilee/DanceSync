class ScoreData:
    def __init__(self):
        """Initializes the ScoreData with an empty dictionary for scores."""
        self.scores = {}  # Dictionary to hold joint scores

    def getScores(self):
        """Returns the dictionary of joint scores.
        
        Returns:
            dict: The scores with joints as keys and their scores as values.
        """
        return self.scores

    def calculateScoreBasedOnAngleDifference(self, angleDifference):
        """Calculates and returns a score based on the angle difference.
        
        Args:
            angleDifference (float): The difference in angles between the reference and student videos.
        
        Returns:
            float: The calculated score based on the angle difference.
        """
        # Placeholder for score calculation logic
        # This could be a simple function or a more complex calculation depending on your scoring criteria.
        # For example, a simple inverse relationship might be used:
        score = max(0, 1 - angleDifference)  # Assuming angleDifference is normalized between 0 and 1
        return score
 