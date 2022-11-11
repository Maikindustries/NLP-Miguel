from transformers import pipeline

class Part1:
    def __init__(self):
        pass

    def run(self):
        lines = []
        with open("tiny_movie_reviews_dataset.txt", "r") as f:
          while True:
            line = f.readline()
            if not line:
                break
            line = line.replace("'", "")
            lines.append(line.strip())
            
        sentiment_pipeline = pipeline("sentiment-analysis")
        for i in range(len(lines)):
          res = sentiment_pipeline(lines[i])
          print(res[0]["label"])