import numpy as np
from random import random
from .targets import RandomRealVector
from .alg_base import Algorithm
from .common_dirty import evalf, copyAttribute, simpleMove
from .patterns import foreach, reducePop, evaluate
import copy


class particleSwarmOptimization(Algorithm):
    def __init__(self, alphabeta=None, gamma=None, delta=None, **kwargs):
        self.opLimitVel = lambda ind, **xt: None
        self.alphabeta = alphabeta
        self.gamma = gamma
        self.delta = delta
        super().__init__(**kwargs)

    @staticmethod
    def updateVel(ind, **xt):
        gamma = evalf(xt['gamma'], inds=[ind], **xt)
        alpha, beta = evalf(xt['alphabeta'], inds=[ind], **xt)
        g = xt['g']
        ind['v'] = gamma * ind['v'] + alpha * (ind['p'] - ind['x']) + beta * (g - ind['x'])

    @staticmethod
    def updateBestPosition(ind, **xt):
        goal = xt['goal']
        if goal.isBetter(ind['fNew'], ind['f']):
            ind['p'] = ind['x'].copy()
            ind['f'] = ind['fNew']

    def start(self):
        super().start("alphabeta gamma g", "&x f *fNew v p")
        foreach(self.population, self.opInit, key='x', **self.env)
        foreach(self.population, copyAttribute, keyFrom='x', keyTo='p', **self.env)
        evaluate(self.population, keyx='x', keyf='f', env=self.env)
        foreach(self.population, copyAttribute, keyFrom='f', keyTo='fNew', **self.env)
        vel = self.delta * (self.target.bounds[1] - self.target.bounds[0])
        foreach(self.population, RandomRealVector((-vel, vel)), key='v', **self.env)

    def runGeneration(self):
        ext = lambda x: (x['p'], x['f'])
        op = lambda x, y: x if self.goal.isBetter(x[1], y[1]) else y
        post = lambda x: x[0]
        self.env['g'] = reducePop(self.population, ext, op, post, timingLabel='reduce')
        foreach(self.population, self.updateVel, timingLabel='updatevel', **self.env)
        foreach(self.population, self.opLimitVel, key='v', timingLabel='limitvel', **self.env)
        foreach(self.population, simpleMove, keyx='x', keyv='v', dt=1.0, timingLabel='move', **self.env)
        evaluate(self.population, keyx='x', keyf='fNew', timingLabel='evaluate', env=self.env)
        foreach(self.population, self.updateBestPosition, timingLabel='updatebest', **self.env)

class randomAlphaBeta:
    def __init__(self, a, b=0):
        self.alpha = a
        self.beta = b if b > 0 else a
    def __call__(self, **xt):
        a = random() * self.alpha
        b = random() * self.beta
        return a, b

class linkedAlphaBeta:
    def __init__(self, t):
        self.total = t
    def __call__(self, **xt):
        a = random() * self.total
        b = self.total - a
        return a, b

class maxAmplitude:
    def __init__(self, amax):
        self.amax = amax
    def __call__(self, ind, **xt):
        key = xt['key']
        a = np.linalg.norm(ind[key])
        amax = evalf(self.amax, inds=[ind], **xt)
        if a > amax:
            ind[key] *= amax / a

class fixedAmplitude:
    def __init__(self, ampl):
        self.ampl = ampl
    def __call__(self, ind, **xt):
        key = xt['key']
        a = np.linalg.norm(ind[key])
        ampl = evalf(self.ampl, inds=[ind], **xt)
        ind[key] *= ampl / a
