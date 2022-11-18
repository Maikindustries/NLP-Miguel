from transformers import pipeline

class SentimentAnalysis:
    def __init__(self, file_name):
        self.text_lines = []
        self.read_file(file_name)

    def read_file(self, file_name):
        if self.text_lines:
            self.text_lines = []
        with open(file_name, "r") as f:
          lines = f.readlines()
           for line in lines: 
            line = line.replace("'", "")
            self.text_lines.append(line.strip())

    def run_analysis(self):
        if not self.text_lines:
            raise Exception("A file must be read first!")
        sentiment_pipeline = pipeline("sentiment-analysis") # nit: think you can do a batch here with a list of lines instead of calling sentiment_pipeline on each one individually! usually that is way more efficient. 
        results = []
        for line in self.text_lines:
            res = sentiment_pipeline(line)
            results.append(res[0]["label"])
        return results


class Part1:
    def run(self):
        sentiment_a = SentimentAnalysis("tiny_movie_reviews_dataset.txt")
        results = sentiment_a.run_analysis()
        for result in results:
            print(result)
