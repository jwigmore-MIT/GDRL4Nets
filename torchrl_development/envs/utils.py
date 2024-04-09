import numpy as np



def create_discrete_rv(rng, nums, probs):
    if probs == 1 and isinstance(nums, int):
        return FakeRV(num=nums)
    elif isinstance(nums, int) and probs < 1:
        return BernRV(rng, num=nums, prob=probs)
    elif isinstance(nums, list):
        return CatRV(rng, nums=nums, probs=probs)


def create_poisson_rv(rng, rate):
    return PoissonRV(rng, rate)

class PoissonRV:

    def __init__(self, rng, rate):
        self.rng = rng
        self.rate = rate

    def sample(self):
        return self.rng.poisson(self.rate)

    def mean(self):
        return self.rate

class FakeRV:
    def __init__(self, num):
        self.num = num

    def sample(self):
        return self.num

    def mean(self):
        return self.num



class BernRV:

    def __init__(self, rng, num = 1, prob = 0.5):
        self.rng = rng
        self.num = num
        self.prob = prob

    def sample(self):
        if self.prob == 1:
            return self.num
        else:
            return int(self.rng.choice([0, self.num], 1, p=[1 - self.prob, self.prob]))

    def mean(self):
        return self.num * self.prob

    def max(self):
        return self.num

    def dist(self):
        # return a discrete distribution as two lists, one for the values and one for the probabilities
        return {0: 1 - self.prob, self.num: self.prob}

class CatRV:

    def __init__(self, rng, nums = [0,1], probs = None):
        self.rng = rng
        self.nums = nums
        self.probs = probs

    def sample(self):
        if self.probs is None:
            return self.nums
        else:
            return int(self.rng.choice(self.nums, 1, p=self.probs))

    def mean(self):
        return np.dot(self.nums, self.probs)

    def max(self):
        return np.max(self.nums)

    def dist(self):
        return {self.nums[i]: self.probs[i] for i in range(len(self.nums))}


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

    @property
    def is_full(self):
        return self.count == self.circularBuffer.buffer_size

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


class TimeAverageStatsCalculator:
    """
    This class is used to calculate the time average statistics of a given buffer size
    We use this to compute the time average statistics of the mean of a long window size
    It allows us to test for convergence of the mean of a long window size
    """

    def __init__(self, buffer_size):
        self.running_stats_calculator = RunningStatsCalculator(buffer_size)
        self.time_average_stats_calculator = RunningStatsCalculator(1000)

    def update(self, new_value):
        self.running_stats_calculator.update(new_value)
        self.time_average_stats_calculator.update(self.running_stats_calculator.mean)

    @property
    def mean(self):
        return self.time_average_stats_calculator.mean

    @property
    def sampleStdev(self):
        return self.time_average_stats_calculator.sampleStdev

    @property
    def is_full(self):
        return self.time_average_stats_calculator.is_full

    def coefVar(self):
        return self.time_average_stats_calculator.sampleStdev / self.time_average_stats_calculator.mean

    def reset(self):
        self.running_stats_calculator = RunningStatsCalculator(self.running_stats_calculator.circularBuffer.buffer_size)
        self.time_average_stats_calculator = RunningStatsCalculator(1000)