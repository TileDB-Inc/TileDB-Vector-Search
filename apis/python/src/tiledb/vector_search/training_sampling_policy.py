import enum


class TrainingSamplingPolicy(enum.Enum):
    FIRST_N = 1
    RANDOM = 2
    SOURCE_URI = 3

    def __str__(self):
        return self.name.replace("_", " ").title()
