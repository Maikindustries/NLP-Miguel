from transformers import pipeline
import unittest

class Part1(unittest.TestCase):
    def run(self):
        lines = []
        with open("tiny_movie_reviews_dataset.txt", "r") as f:
          while True:
            line = f.readline()
            if not line:
                break
            line = line.replace("'", "")
            lines.append(line.strip())
            
        results = []
        sentiment_pipeline = pipeline("sentiment-analysis")
        for i in range(len(lines)):
            res = sentiment_pipeline(lines[i])
            results.append(res[0]["label"])
            print(res[0]["label"])
        expected_results = [
            "NEGATIVE",
            "POSITIVE",
            "POSITIVE",
            "NEGATIVE",
            "NEGATIVE",
            "POSITIVE",
            "NEGATIVE",
            "POSITIVE",
            "NEGATIVE",
            "POSITIVE",
            "POSITIVE",
            "POSITIVE",
            "NEGATIVE",
            "NEGATIVE",
            "POSITIVE",
            "POSITIVE",
            "POSITIVE",
            "POSITIVE",
            "POSITIVE",
            "NEGATIVE"
        ]
        
        # Test
        self.assertEqual(expected_results, results)
