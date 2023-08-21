import numpy as np


class Noise:
    def __init__(self, mu=None, sigma=0.05, theta=.25, dimension=1e-2, x0=None, num_steps=12000, size=1):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dimension
        self.x0 = x0
        self.size = size

    def step(self):
        pass

    def reset(self):
        pass


class OrnsteinUhlenbeckActionNoise(Noise):
    def __init__(self, mu=None, sigma=0.05, theta=.25, dimension=1e-2, x0=None, num_steps=12000, size=1):
        super(OrnsteinUhlenbeckActionNoise, self).__init__(
            mu, sigma, theta, dimension, x0, num_steps, size
        )
        self.reset()

    def step(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class GausianNoise(Noise):
    def step(self, step=None):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=self.size)

