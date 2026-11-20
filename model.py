import numpy as np

LEVELS = np.linspace(0.05, 0.95, 4)
FEATURES = np.array([(a, b) for a in LEVELS for b in LEVELS])      # (16, 2)


def feature_vector(f):
    return np.repeat(FEATURES[f], 3)                                # n = 6


def object_vector(feats):
    return np.concatenate([feature_vector(f) for f in feats])      # n = 24


class KohonenGrid:
    def __init__(self, input_dim, size=200, seed=None, learning_rate=0.35):
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weights = self.rng.random((size, size, input_dim))

    def winner(self, stimulus):
        d = ((self.weights - stimulus) ** 2).sum(axis=2)
        return np.unravel_index(np.argmin(d), d.shape)

    def encode(self, stimulus, cycles=20, learning_rate=None, neighborhood=2.085):
        lr = self.learning_rate if learning_rate is None else learning_rate
        stimulus = np.asarray(stimulus, dtype=float)
        R = min(int(np.ceil(3 * neighborhood)), self.size // 2 - 1)   # v < ~1e-4 beyond 3G
        off = np.arange(-R, R + 1)
        r = np.abs(off)[:, None] + np.abs(off)[None, :]               # city-block distance
        profile = np.exp(-(r / neighborhood) ** 2)[..., None]
        for _ in range(cycles):
            d = ((self.weights - stimulus) ** 2).sum(axis=2)
            wy, wx = np.unravel_index(np.argmin(d), d.shape)
            ix = np.ix_((wy + off) % self.size, (wx + off) % self.size)
            patch = self.weights[ix]
            self.weights[ix] = patch + lr * profile * (stimulus - patch)

    def selectivity(self, stimulus):
        dist = np.sqrt(((self.weights - stimulus) ** 2).sum(axis=2) / self.input_dim)
        wy, wx = np.unravel_index(np.argmin(dist), dist.shape)
        a = np.log(1.0 / np.maximum(dist, 1e-4))                     # cap = ln(1e4) = 9.210
        rows = [(wy + i) % self.size for i in (-1, 0, 1)]
        cols = [(wx + j) % self.size for j in (-1, 0, 1)]
        return a[np.ix_(rows, cols)].sum() / a.sum()


class PMFCNetwork:
    def __init__(self, seed=0, lesioned=False, size=200):
        self.size = size
        self.lesioned = lesioned
        self.rng_c = np.random.default_rng(seed * 100 + 50)      # caudal pretraining stimuli
        self.rng_p = np.random.default_rng(seed * 100 + 51)      # PRh pretraining stimuli
        self.rng_stim = np.random.default_rng(seed * 100 + 52)   # interfering stimuli
        self.caudal = [KohonenGrid(6, size, seed=seed * 100 + i) for i in range(4)]
        self.perirhinal = None if lesioned else KohonenGrid(24, size, seed=seed * 100 + 4)

    def grids(self):
        return self.caudal + ([self.perirhinal] if self.perirhinal is not None else [])

    def pretrain(self, cycles=100):
        for t in range(1, cycles + 1):
            alpha = t ** -0.2                       # A = 0.2
            G = 0.5 + 10 * t ** -0.4                # B = 0.4
            for grid in self.caudal:
                f = int(self.rng_c.integers(0, 16))
                grid.encode(feature_vector(f), 1, alpha, G)
            if self.perirhinal is not None:
                obj = tuple(int(x) for x in self.rng_p.integers(0, 16, size=4))
                self.perirhinal.encode(object_vector(obj), 1, alpha, G)
        for g in self.grids():
            g.weights += g.rng.uniform(-1, 1, g.weights.shape)
            g.weights = (g.weights + 1) / 3

    def encode_object(self, obj, cycles=20):
        for i, g in enumerate(self.caudal):
            g.encode(feature_vector(obj[i]), cycles, 0.35, 2.085)
        if self.perirhinal is not None:
            self.perirhinal.encode(object_vector(obj), cycles, 0.35, 2.085)

    def interfere(self, count, alpha=0.05):
        for _ in range(count):
            obj = tuple(int(x) for x in self.rng_stim.integers(0, 16, size=4))
            for i, g in enumerate(self.caudal):
                g.encode(feature_vector(obj[i]), 1, alpha, 2.085)
            if self.perirhinal is not None:
                self.perirhinal.encode(object_vector(obj), 1, alpha, 2.085)

    def scores(self, sample, novel):
        cs = np.mean([g.selectivity(feature_vector(sample[i])) for i, g in enumerate(self.caudal)])
        cn = np.mean([g.selectivity(feature_vector(novel[i])) for i, g in enumerate(self.caudal)])
        r_cau = (cs - cn) / (cs + cn)
        if self.perirhinal is None:
            return r_cau, r_cau
        ps = self.perirhinal.selectivity(object_vector(sample))
        pn = self.perirhinal.selectivity(object_vector(novel))
        r_prh = (ps - pn) / (ps + pn)
        return r_cau, (r_cau + r_prh) / 2

    def snapshot(self):
        return [g.weights.copy() for g in self.grids()]

    def restore(self, snap):
        for g, w in zip(self.grids(), snap):
            g.weights[...] = w
