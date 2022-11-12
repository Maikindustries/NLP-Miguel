from transformers import pipeline

class SentimentAnalysis:
    def __init__(self, file_name):
        self.text_lines = []
        self.read_file(file_name)

    def read_file(self, file_name):
        if self.text_lines:
            self.text_lines = []
        with open(file_name, "r") as f:
          while True:
            line = f.readline()
            if not line:
                break
            line = line.replace("'", "")
            self.text_lines.append(line.strip())

    def run_analysis(self):
        if not self.text_lines:
            raise Exception("A file must be read first!")
        sentiment_pipeline = pipeline("sentiment-analysis")
        results = []
        for i in range(len(self.text_lines)):
            res = sentiment_pipeline(self.text_lines[i])
            results.append(res[0]["label"])
        return results


class Part1:
    def run(self):
        sentiment_a = SentimentAnalysis("tiny_movie_reviews_dataset.txt")
        results = sentiment_a.run_analysis()
        for result in results:
            print(result)
