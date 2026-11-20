import unittest
import numpy as np
from model import PMFCNetwork, KohonenGrid, feature_vector, object_vector


class PMFCTests(unittest.TestCase):
    def test_architecture(self):
        c = PMFCNetwork(seed=1, size=40)
        l = PMFCNetwork(seed=1, size=40, lesioned=True)
        self.assertEqual(len(c.caudal), 4)
        self.assertIsNotNone(c.perirhinal)
        self.assertIsNone(l.perirhinal)

    def test_dimensions(self):
        self.assertEqual(feature_vector(0).shape, (6,))
        self.assertEqual(object_vector((0, 1, 2, 3)).shape, (24,))

    def test_G_schedule(self):
        self.assertAlmostEqual(0.5 + 10 * 100 ** -0.4, 2.085, places=3)

    def test_encoding_sharpens(self):
        g = KohonenGrid(6, size=40, seed=0)
        s = feature_vector(3)
        before = g.selectivity(s)
        g.encode(s, 20, 0.35, 2.085)
        self.assertGreater(g.selectivity(s), before)

    def test_identical_pair_is_zero(self):
        net = PMFCNetwork(seed=1, size=40)
        net.pretrain()
        les, ctl = net.scores((0, 1, 2, 3), (0, 1, 2, 3))
        self.assertAlmostEqual(les, 0.0)
        self.assertAlmostEqual(ctl, 0.0)

    def test_restore_exact(self):
        net = PMFCNetwork(seed=1, size=40)
        net.pretrain()
        snap = net.snapshot()
        net.encode_object((0, 1, 2, 3))
        net.restore(snap)
        for g, w in zip(net.grids(), snap):
            self.assertTrue(np.array_equal(g.weights, w))

    def test_lesion_leaves_caudal_unchanged(self):
        a = PMFCNetwork(seed=1, size=40)
        b = PMFCNetwork(seed=1, size=40, lesioned=True)
        a.pretrain()
        b.pretrain()
        for ga, gb in zip(a.caudal, b.caudal):
            self.assertTrue(np.array_equal(ga.weights, gb.weights))


if __name__ == "__main__":
    unittest.main()
