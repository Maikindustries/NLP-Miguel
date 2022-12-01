from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

PERCENT_OF_DATASET_TO_TRAIN = 0.2
NER_MODEL_FAST = "flair/ner-english-ontonotes-fast"
NER_MODEL = "resources/taggers/ner-english"
FINAL_MODEL = "resources/taggers/ner-english/final-model.pt"
MODEL_LOSS = "resources/taggers/ner-english/loss.tsv"

class NERModel:
  def __init__(self):
    self.corpus = []
    self.model = None

  def get_corpus(self):
    columns = {0 : 'text',
              1 : 'ner'}
    data_folder = 'content'
    corpus: Corpus = ColumnCorpus(data_folder, 
                                  columns,
                                  train_file='train',
                                  test_file='test',
                                  dev_file='dev')
    self.corpus = corpus.downsample(PERCENT_OF_DATASET_TO_TRAIN)
    return self.corpus
    

  def train(self):
    if self.corpus == []:
      raise Exception("A file must be read first!")
    tagger = SequenceTagger.load(NER_MODEL_FAST)
    trainer = ModelTrainer(tagger, self.corpus)
    trainer.train(NER_MODEL, max_epochs=10)
  
  def get_model(self):
    self.model = SequenceTagger.load(FINAL_MODEL)
    return self.model

  def plot(self):
    plotter = Plotter()
    plotter.plot_training_curves(MODEL_LOSS)


class Part2:
  def run(self):
    self.get_Corpus()
    self.train()
    self.get_model()
    self.plot()