
class BaseTokenizer(object):
    def __init__(self, *args, **kwargs):
        pass

    def tokenize(self, text):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, tokens):
        raise NotImplementedError

    def __call__(self, text):
        return self.tokenize(text)

    def __repr__(self):
        return self.__class__.__name__
