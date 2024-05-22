from abc import abstractmethod

class Agent:
    @abstractmethod
    def reset(self, verbose, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, dataset, verbose, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError
    
    @abstractmethod
    def save(self, filename):
        raise NotImplementedError

    @abstractmethod
    def load(self, filename):
        raise NotImplementedError