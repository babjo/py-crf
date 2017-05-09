class FeatureFunction(object):
    def __init__(self, previous_label, label):
        self.previous_label = previous_label
        self.label = label

    def apply(self, previous_label, label):
        return 1 if self.previous_label == previous_label and self.label == label else 0

    def __hash__(self):
        return hash(self.previous_label) ^ hash(self.label)

    def __eq__(self, other):
        return (self.previous_label, self.label) == (other.previous_label, other.label)
