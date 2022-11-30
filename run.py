from src.Part1 import Part1
from src.Part3 import Part3
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter

"""
  Author:
    Miguel Ángel Pérez López A01750145
"""

def main():
  # Part 1
  part1 = Part1()
  part1.run()

  # Part 2
  # Nit: put this in a Part2 file/class for consistency with the others
  PERCENT_OF_DATASET_TO_TRAIN = 0.2 # constants should go at top of module! 
  columns = {0 : 'text',
             1 : 'ner'}
  data_folder = 'content'
  corpus: Corpus = ColumnCorpus(data_folder, 
                                columns,
                                train_file='train',
                                test_file='test',
                                dev_file='dev')

  corpus = corpus.downsample(PERCENT_OF_DATASET_TO_TRAIN)
  tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast") #nit: usually paths are pulled out as constants at top of module 
  trainer = ModelTrainer(tagger, corpus)
  trainer.train('resources/taggers/ner-english', max_epochs=10) # same re: constants
  model = SequenceTagger.load('resources/taggers/ner-english/final-model.pt')
  # Graph
  plotter = Plotter()
  plotter.plot_training_curves('resources/taggers/ner-english/loss.tsv')
  
  # Part 3
  part3 = Part3()
  part3.run()


if __name__ == '__main__':
  main()
