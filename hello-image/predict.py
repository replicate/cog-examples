from cog import BasePredictor, Path

class Predictor(BasePredictor):
    def predict(self) -> Path:
        return Path("hello.webp")
