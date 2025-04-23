class Ggd:
    def __init__(self, metadata, data):
        self.data = data
        self.metadata = metadata

    def __call__(self):
        return self.data
    
    def metadata(self):
        return self.metadata
    
    def data(self):
        return self.data
