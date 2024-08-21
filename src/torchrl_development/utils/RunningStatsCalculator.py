import numpy as np


class CircularBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer = []

    def append(self, value):
        popped_value = None
        if len(self.buffer) == self.buffer_size:
            popped_value = self.buffer[self.index]
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)  # Expand buffer if not full yet
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.buffer_size
        return popped_value

    @property
    def length(self):
        return len(self.buffer)

    def items(self):
        return self.buffer

class RunningStatsCalculator:
    def __init__(self, buffer_size):
        self.circularBuffer = CircularBuffer(buffer_size)
        self._mean = 0
        self._dSquared = 0

    @property
    def count(self):
        return self.circularBuffer.length

    def update(self, new_value):
        popped_value = self.circularBuffer.append(new_value)

        if self.count == 1 and popped_value is None:
            # initialize when the first value is added
            self._mean = new_value
            self._dSquared = 0
        elif popped_value is None:
            # if the buffer is not full yet, use standard Welford method
            mean_increment = (new_value - self._mean) / self.count
            new_mean = self._mean + mean_increment

            d_squared_increment = ((new_value - new_mean) * (new_value - self._mean))
            new_d_squared = self._dSquared + d_squared_increment

            self._mean = new_mean
            self._dSquared = new_d_squared
        else:
            # once the buffer is full, adjust Welford Method for window size
            mean_increment = (new_value - popped_value) / self.count
            new_mean = self._mean + mean_increment

            d_squared_increment = ((new_value - popped_value) * (new_value - new_mean + popped_value - self._mean))
            new_d_squared = self._dSquared + d_squared_increment

            self._mean = new_mean
            self._dSquared = new_d_squared

    @property
    def mean(self):
        return self._mean

    @property
    def dSquared(self):
        return self._dSquared

    @property
    def populationVariance(self):
        return self.dSquared / self.count

    @property
    def populationStdev(self):
        return np.sqrt(self.populationVariance)

    @property
    def sampleVariance(self):
        return self.dSquared / (self.count - 1) if self.count > 1 else 0

    @property
    def sampleStdev(self):
        return np.sqrt(self.sampleVariance)