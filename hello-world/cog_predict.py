import cog


class Model(cog.Model):
    def setup(self):
        self.prefix = "hello"

    @cog.input("text", type=str, help="Text that will get prefixed by 'hello '")
    def predict(self, text):
        return self.prefix + " " + text
