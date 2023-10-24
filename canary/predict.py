from cog import BasePredictor, ConcatenateIterator, Input


class Predictor(BasePredictor):
    def predict(self, text: str = Input(description="Text to prefix with 'hello there, '")) -> ConcatenateIterator[str]:
        yield "hello "
        yield "there, "
        yield text
