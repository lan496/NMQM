import unittest

from hydrogen_radial import SphericalSchrodingerSolver


class TestSphericalSchrodinger(unittest.TestCase):

    def setUp(self):
        pass

    def test_hydrogen(self):
        atomic_charge = 2
        sss = SphericalSchrodingerSolver(atomic_charge)
        for n in range(1, 6):
            for l in range(0, n):
                energy_actual = sss.solve(n, l)
                energy_expected = - (atomic_charge / n) ** 2
                print(f"n={n}, l={l}: E={energy_actual:.4f} Ry (expected={energy_expected:.4f} Ry)")
                self.assertAlmostEqual(energy_actual, energy_expected, places=4)


if __name__ == '__main__':
    unittest.main()
