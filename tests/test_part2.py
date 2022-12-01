import unittest
import sys
import os
 
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from src.Part2 import NERModel

class test_part2(unittest.TestCase):
  def test_1(self):
    ner_model = NERModel()
    self.assertEqual(ner_model.corpus, None)
    self.assertEqual(ner_model.get_corpus() is not None, True)
    