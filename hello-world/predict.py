import cog


class Predictor(cog.Predictor):
    def setup(self):
        self.prefix = "hello"

    @cog.input("input", type=str, help="Text that will get prefixed by 'hello '")
    def predict(self, input):
        return self.prefix + " " + input
