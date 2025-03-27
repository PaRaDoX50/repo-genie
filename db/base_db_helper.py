class BaseDBHelper():
    def __init__(self):
        pass
        
    def get(self, query, params=None):
        return NotImplementedError()

    def save(self, query, params=None):
        return NotImplementedError()