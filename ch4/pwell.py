import numpy as np


class PWPotential:

    def __init__(self, width, V0):
        self.width = width
        self.V0 = V0
        assert(V0 > 0)

    def _get_wave_number(self, n_planewave, period):
        kn = np.zeros(n_planewave)
        kn[1::2] = np.array(range(1, n_planewave // 2 + 1)) * 2 * np.pi / period
        kn[2::2] = -kn[1::2]
        return kn

    def get_hamiltonian_matrix(self, n_planewave, period):
        kn = self._get_wave_number(n_planewave, period)
        H = np.zeros((n_planewave, n_planewave))

        for i in range(n_planewave):
            for j in range(n_planewave):
                if i == j:
                    H[i, j] = kn[i] ** 2 - self.V0 * self.width / period
                else:
                    theta = self.width * (kn[i] - kn[j]) / (2. * np.pi)
                    H[i, j] = -self.V0 * self.width / period * np.sinc(theta)

        return H


class Variation:
    """
    Calculating ground state energy via plane-wave expansion.
    We use Rydberg atomic unit.
    """

    def __init__(self, n, period):
        self.n = n
        assert(self.n > 0)
        self.period = period
        assert(self.period > 0)

        self.n_planewave = 2 * self.n + 1

    def solve(self, pwpot: PWPotential):
        H = pwpot.get_hamiltonian_matrix(self.n_planewave, self.period)
        eigvals, eigvecs = np.linalg.eigh(H)  # in ascending order

        return eigvals


if __name__ == '__main__':
    pwpot = PWPotential(width=2., V0=1.)
    var = Variation(n=300, period=8)
    eigvals = var.solve(pwpot)
    print(eigvals[:3])
    assert(np.isclose(eigvals[0], -0.4538))
