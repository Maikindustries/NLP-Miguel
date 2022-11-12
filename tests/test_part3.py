import unittest
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from src.Part3 import Translators

class test_part3(unittest.TestCase):
  def test_1(self):
    translators = Translators("es.txt","en.txt")
    google_avg, helsinki_avg = translators.translate()
    self.assertEqual(google_avg, 0.32662462131108)
    self.assertEqual(helsinki_avg, 0.31194402697495693)