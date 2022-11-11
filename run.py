from Part1 import Part1
from Part3 import Part3

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
import inspect
from flair.visual.training_curves import Plotter

def main():
  # Part 1
  part1 = Part1()
  part1.run()

  print()

  # Part 2
  columns = {0 : 'text', 1 : 'ner'}
  data_folder = '/content/'
  corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file = 'train',
                                test_file = 'test',
                                dev_file = 'dev')

  corpus = corpus.downsample(0.2)
  # tag_dictionary = corpus.make_label_dictionary(label_type = "ner")

  # tag_dictionary = corpus.make_label_dictionary(label_type = "ner")
  tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")

  trainer = ModelTrainer(tagger, corpus)

  trainer.train('resources/taggers/ner-english',train_with_dev=True,max_epochs=25,
                monitor_train = True, monitor_test = True)

  model = SequenceTagger.load('resources/taggers/ner-english/final-model.pt')

  # num = 6
  # exp = corpus.test[num].to_tagged_string('text')
  # print(corpus.test[num].to_tagged_string('ner'))
  # sentence = Sentence(exp)

  # model.predict(sentence)

  # for entity in sentence.get_spans('ner'):
  #     print(entity)
  
  # print(inspect.signature(trainer.train))
  
  plotter = Plotter()
  plotter.plot_training_curves('/content/resources/taggers/ner-english/loss.tsv')
  
  # Part 3
  print()
  part3 = Part3()
  part3.run()


if __name__ == '__main__':
  main()