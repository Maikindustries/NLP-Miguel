from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline
import unittest

class Part3(unittest.TestCase):
  def run(self):
    # read spanish sentences
    lines_es = []
    with open("es.txt", "r") as f:
      while True:
        line = f.readline()
        if not line:
            break
        line = line.replace(".", "")
        lines_es.append(line.strip())

    # read english sentences
    lines_en = []
    with open("en.txt", "r") as f:
      while True:
        line = f.readline()
        if not line:
            break
        line = line.replace(".", "")
        lines_en.append(line.strip())

    #Google translator
    translator_g = Translator()

    # Helsinki-NLP translator
    model_checkpoint = "Helsinki-NLP/opus-mt-es-en"
    translator_h = pipeline("translation", model=model_checkpoint)

    results_g, results_h = [], []
    for es, en in zip(lines_es, lines_en):
      # Google
      translation = translator_g.translate(es, src="es", dest="en")
      poss_translations = translation.extra_data["possible-translations"][0][2]
      refs = []
      for i in poss_translations:
        refs.append(i[0].split())
      bleu_score = sentence_bleu(refs, en.split())
      results_g.append(bleu_score)

      # Helsinki
      translation = translator_h(es)[0]["translation_text"].split()
      bleu_score = sentence_bleu([translation], en.split())
      results_h.append(bleu_score)

    # Calculate the average
    google_avg = sum(results_g) / len(results_g)
    helsinki_avg = sum(results_h) / len(results_h)
    print("GOOGLE_TRANSLATOR ", google_avg)
    print("Helsinki-NLP/opus-mt-es-en ", helsinki_avg)

    # Tests
    self.assertEqual(google_avg, 0.32662462131108)
    self.assertEqual(helsinki_avg, 0.31194402697495693)
