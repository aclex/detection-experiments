class FeatureHook:
    def __init__(self):
        self.output = None

    def __call__(self, module, input, output):
        self.output = output
