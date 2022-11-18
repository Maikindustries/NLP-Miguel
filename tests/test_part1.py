import unittest
import sys
import os
 # Add tests for other parts! When writing code, I write the test for each class or piece of functionality as soon as I finish that piece. It helps you develop incrementally, being sure that each piece of code is clean and works like you expect it to! 

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from src.Part1 import SentimentAnalysis

class test_part1(unittest.TestCase):
  def test_1(self):
    sentiment_a = SentimentAnalysis("../src/tiny_movie_reviews_dataset.txt")
    results = sentiment_a.run_analysis()
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
    self.assertEqual(expected_results, results)
