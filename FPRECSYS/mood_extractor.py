from typing import Dict

class MoodExtractor:
    """Extract mood adjustments from user input"""
    
    def __init__(self):
        self.mood_rules = {
            "energetic": {"energy": 0.7, "danceability": 0.6, "valence": 0.4, "tempo": 0.5},
            "sad": {"valence": -0.8, "energy": -0.5, "acousticness": 0.3},
            "acoustic": {"acousticness": 0.7, "energy": -0.3, "instrumentalness": 0.2},
            "chill": {"energy": -0.6, "tempo": -0.5, "acousticness": 0.4},
            "dance": {"danceability": 0.8, "energy": 0.7, "tempo": 0.5},
            "electronic": {"acousticness": -0.8, "energy": 0.6, "instrumentalness": 0.5},
            "upbeat": {"energy": 0.7, "valence": 0.7, "tempo": 0.4},
            "melancholic": {"valence": -0.7, "energy": -0.4, "acousticness": 0.5},
            "relaxing": {"energy": -0.7, "valence": 0.3, "loudness": -0.4},
            "happy": {"valence": 0.8, "energy": 0.5, "danceability": 0.5},
            "slow": {"tempo": -0.6, "energy": -0.4},
            "fast": {"tempo": 0.6, "energy": 0.5},
            "loud": {"loudness": 0.5, "energy": 0.4},
            "quiet": {"loudness": -0.6, "energy": -0.3},
            "instrumental": {"instrumentalness": 0.8, "speechiness": -0.5},
            "vocal": {"instrumentalness": -0.6, "speechiness": 0.4}
        }
    
    def extract(self, user_message: str) -> Dict[str, float]:
        """Extract mood adjustments from user message"""
        msg = user_message.lower()
        
        adjustments = {
            "energy": 0, "danceability": 0, "valence": 0, "acousticness": 0,
            "instrumentalness": 0, "speechiness": 0, "liveness": 0, "loudness": 0, "tempo": 0
        }
        
        for keyword, values in self.mood_rules.items():
            if keyword in msg:
                adjustments.update(values)
        
        return adjustments
