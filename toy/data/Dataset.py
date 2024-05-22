import pickle as pkl

class Dataset:
    def __init__(self):
        self.size = 0
        self.data = None

    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self.data, f)
    
    def load(self, filename):
        with open(filename, "rb") as f:
            self.data = pkl.load(f)
    
    def store(self, data):
        self.data = data
        self.size = len(data)