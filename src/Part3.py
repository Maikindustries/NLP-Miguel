from googletrans import Translator
from nltk.translate.bleu_score import sentence_bleu
from transformers import pipeline

class Translators:
  def __init__(self, es_file_name, en_file_name):
    self.es_text_lines = []
    self.en_text_lines = []
    self.read_files(es_file_name, en_file_name)

  def read_files(self, es_file_name, en_file_name):
    # read spanish sentences
    with open(es_file_name, "r") as f:
      lines = f.readlines(0
      self.es_text_lines = [line.replace(".", "").strip() for line in lines]

    # read english sentences
    with open(en_file_name, "r") as f:
      while True: # same readlines edit as above
        line = f.readline()
        if not line:
            break
        line = line.replace(".", "")
        self.en_text_lines.append(line.strip())

  def translate(self):
    #Google translator
    translator_g = Translator()

    # Helsinki-NLP translator
    model_checkpoint = "Helsinki-NLP/opus-mt-es-en"
    translator_h = pipeline("translation", model=model_checkpoint)

    results_g, results_h = [], []
    for es, en in zip(self.es_text_lines, self.en_text_lines):
      # Google
      translation = translator_g.translate(es, src="es", dest="en")
      poss_translations = translation.extra_data["possible-translations"][0][2]
      refs = []
      for poss_translation in poss_translations:
        refs.append(poss_translation[0].split())
      bleu_score = sentence_bleu(refs, en.split())
      results_g.append(bleu_score)

      # Helsinki
      translation = translator_h(es)[0]["translation_text"].split()
      bleu_score = sentence_bleu([translation], en.split())
      results_h.append(bleu_score)

    # Calculate the average
    google_avg = sum(results_g) / len(results_g) # nit: you'll get divide by 0 if no lines...add an assert before to check that! 
    helsinki_avg = sum(results_h) / len(results_h)
    return google_avg, helsinki_avg


class Part3:
  def run(self):
    translators = Translators("es.txt","en.txt")
    google_avg, helsinki_avg = translators.translate()
    print(google_avg)
    print(helsinki_avg)
