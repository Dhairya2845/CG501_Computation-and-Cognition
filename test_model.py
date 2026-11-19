import unittest

from model import PMFCNetwork


class PMFCTests(unittest.TestCase):
    def test_control_has_perirhinal_layer(self):
        network = PMFCNetwork(seed=1, lesioned=False)
        self.assertIsNotNone(network.perirhinal)
        self.assertEqual(len(network.caudal), 4)

    def test_lesion_removes_perirhinal_layer(self):
        network = PMFCNetwork(seed=1, lesioned=True)
        self.assertIsNone(network.perirhinal)
        self.assertEqual(len(network.caudal), 4)

    def test_objects_are_eight_dimensional(self):
        network = PMFCNetwork(seed=1)
        pair = network.make_features if hasattr(network, "make_features") else None
        object_vector = network.object_vector((0, 1, 2, 3))
        self.assertEqual(object_vector.shape, (8,))


if __name__ == "__main__":
    unittest.main()
