import unittest

from harmonic import solve1D


class TestNumerov(unittest.TestCase):

    def setUp(self):
        self.xmax = 10.
        self.mesh = 100
        self.n_iter = 1000

    def test_harmonic(self):
        for nodes in range(1, 10 + 1):
            x, y, eigval = solve1D(self.xmax, self.mesh, nodes, self.n_iter)
            expected = nodes + 0.5
            self.assertAlmostEqual(eigval, expected, places=3)
            print(f"nodes={nodes}: actual({eigval:.8f}), expected({expected:.8f})")


if __name__ == '__main__':
    unittest.main()
