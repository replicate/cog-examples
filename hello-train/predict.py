import requests
from cog import BasePredictor, Input, Path

from typing import Optional


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        if weights:
            self.prefix = requests.get(weights).text
        else:
            self.prefix = "hello"

    def predict(
        self, text: str = Input(description="Text to prefix with 'hello ' or weights")
    ) -> str:
        return self.prefix + " " + text
