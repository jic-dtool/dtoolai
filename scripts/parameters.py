
class Parameters(object):

    def __init__(self):

        self.parameter_dict = {}

    def __getattr__(self, name):
        return self.parameter_dict[name]
