import numpy as np


class KohonenGrid:
    def __init__(self, input_dim, size=20, seed=None, learning_rate=0.35):
        rng = np.random.default_rng(seed)
        self.size = size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        yy, xx = np.indices((size, size))
        self.locations = np.stack((yy, xx), axis=-1)
        self.weights = rng.random((size, size, input_dim))

    def copy(self):
        grid = KohonenGrid(self.input_dim, self.size)
        grid.weights = self.weights.copy()
        grid.learning_rate = self.learning_rate
        return grid

    def winner(self, stimulus):
        distances = np.linalg.norm(self.weights - stimulus, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def encode(self, stimulus, cycles=20, learning_rate=None, neighborhood=2.085):
        if learning_rate is None:
            learning_rate = self.learning_rate
        stimulus = np.asarray(stimulus, dtype=float)
        for _ in range(cycles):
            winner = self.winner(stimulus)
            wy, wx = winner
            dy = np.minimum(
                np.abs(self.locations[:, :, 0] - wy),
                self.size - np.abs(self.locations[:, :, 0] - wy),
            )
            dx = np.minimum(
                np.abs(self.locations[:, :, 1] - wx),
                self.size - np.abs(self.locations[:, :, 1] - wx),
            )
            r = np.sqrt(dy * dy + dx * dx)
            profile = np.exp(-(r * r) / (neighborhood * neighborhood))
            delta = learning_rate * profile[:, :, None] * (
                stimulus[None, None, :] - self.weights
            )
            self.weights += delta

    def selectivity(self, stimulus):
        distances = np.linalg.norm(self.weights - stimulus, axis=2)
        activation = np.exp(-distances)
        peak_unit = np.unravel_index(np.argmin(distances), distances.shape)
        wy, wx = peak_unit
        dy = np.minimum(
            np.abs(self.locations[:, :, 0] - wy),
            self.size - np.abs(self.locations[:, :, 0] - wy),
        )
        dx = np.minimum(
            np.abs(self.locations[:, :, 1] - wx),
            self.size - np.abs(self.locations[:, :, 1] - wx),
        )
        k = (dy <= 1) & (dx <= 1)
        k |= (dy <= 1) & (dx == self.size - 1)
        k |= (dy == self.size - 1) & (dx <= 1)
        k |= (dy == self.size - 1) & (dx == self.size - 1)
        return activation[k].sum() / activation.sum()


class PMFCNetwork:
    def __init__(self, seed=0, lesioned=False, size=20):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.lesioned = lesioned
        self.caudal = [
            KohonenGrid(2, size=size, seed=rng.integers(0, 1e9))
            for _ in range(4)
        ]
        self.perirhinal = None if lesioned else KohonenGrid(
            8, size=size, seed=rng.integers(0, 1e9)
        )
        self.features = self.make_features(rng)

    @staticmethod
    def make_features(rng):
        # Each visual feature is a distinct two-dimensional combination.
        features = {}
        for index in range(16):
            features[index] = rng.dirichlet(np.ones(2))
        return features

    def feature_vector(self, feature_id):
        return self.features[feature_id]

    def object_vector(self, object_features):
        return np.concatenate(
            [self.feature_vector(feature) for feature in object_features]
        )

    def pretrain(self, cycles=100):
        for _ in range(cycles):
            for grid_index, grid in enumerate(self.caudal):
                feature_id = int(self.rng.integers(0, 16))
                grid.encode(
                    self.feature_vector(feature_id),
                    cycles=1,
                    learning_rate=0.35,
                    neighborhood=2.085,
                )
            if self.perirhinal is not None:
                object_features = tuple(
                    int(x) for x in self.rng.integers(0, 16, size=4)
                )
                self.perirhinal.encode(
                    self.object_vector(object_features),
                    cycles=1,
                    learning_rate=0.35,
                    neighborhood=2.085,
                )

    def encode_object(self, object_features, cycles=20):
        for grid_index, grid in enumerate(self.caudal):
            grid.encode(
                self.feature_vector(object_features[grid_index]),
                cycles=cycles,
                learning_rate=0.35,
                neighborhood=2.085,
            )
        if self.perirhinal is not None:
            self.perirhinal.encode(
                self.object_vector(object_features),
                cycles=cycles,
                learning_rate=0.35,
                neighborhood=2.085,
            )

    def interfere(self, count):
        for _ in range(count):
            object_features = tuple(int(x) for x in self.rng.integers(0, 16, size=4))
            for grid_index, grid in enumerate(self.caudal):
                grid.encode(
                    self.feature_vector(object_features[grid_index]),
                    cycles=1,
                    learning_rate=0.05,
                    neighborhood=2.085,
                )
            if self.perirhinal is not None:
                self.perirhinal.encode(
                    self.object_vector(object_features),
                    cycles=1,
                    learning_rate=0.05,
                    neighborhood=2.085,
                )

    def recognition_score(self, sample, novel):
        caudal_sample = [
            grid.selectivity(self.feature_vector(sample[i]))
            for i, grid in enumerate(self.caudal)
        ]
        caudal_novel = [
            grid.selectivity(self.feature_vector(novel[i]))
            for i, grid in enumerate(self.caudal)
        ]
        score_caudal = (
            np.mean(caudal_sample) - np.mean(caudal_novel)
        ) / max(np.mean(caudal_sample) + np.mean(caudal_novel), 1e-12)
        if self.perirhinal is None:
            return float(score_caudal)
        prh_sample = self.perirhinal.selectivity(self.object_vector(sample))
        prh_novel = self.perirhinal.selectivity(self.object_vector(novel))
        score_prh = (prh_sample - prh_novel) / max(
            prh_sample + prh_novel, 1e-12
        )
        return float((score_caudal + score_prh) / 2.0)
