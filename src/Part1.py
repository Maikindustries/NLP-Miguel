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
        sentiment_pipeline = pipeline("sentiment-analysis")
        analyzed_text = sentiment_pipeline(self.text_lines)
        results = [result["label"] for result in analyzed_text]
        return results


class Part1:
    def run(self):
        sentiment_a = SentimentAnalysis("tiny_movie_reviews_dataset.txt")
        results = sentiment_a.run_analysis()
        for result in results:
            print(result)