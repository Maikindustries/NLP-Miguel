from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

PERCENT_OF_DATASET_TO_TRAIN = 0.2
NER_MODEL_FAST = "flair/ner-english-ontonotes-fast"
NER_MODEL = "src/resources/taggers/ner-english"
FINAL_MODEL = "src/resources/taggers/ner-english/final-model.pt"
MODEL_LOSS = "src/resources/taggers/ner-english/loss.tsv"

class NERModel:
  def __init__(self):
    self.corpus = None
    self.model = None

  def get_corpus(self):
    columns = {0 : 'text',
               1 : 'ner'}
    data_folder = 'src/content'
    corpus: Corpus = ColumnCorpus(data_folder, 
                                  columns,
                                  train_file='train',
                                  test_file='test',
                                  dev_file='dev')
    self.corpus = corpus.downsample(PERCENT_OF_DATASET_TO_TRAIN)
    return self.corpus

  def train(self):
    if not self.corpus:
      raise Exception("You should train first!")
    tagger = SequenceTagger.load(NER_MODEL_FAST)
    trainer = ModelTrainer(tagger, self.corpus)
    trainer.train(NER_MODEL, max_epochs=15)
  
  def get_model(self):
    self.model = SequenceTagger.load(FINAL_MODEL)

  def plot(self):
    plotter = Plotter()
    plotter.plot_training_curves(MODEL_LOSS)


class Part2:
  def run(self):
    ner_model = NERModel()
    ner_model.get_corpus()
    ner_model.train()
    ner_model.get_model()
    ner_model.plot()