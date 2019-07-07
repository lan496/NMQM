import numpy as np
from scipy.linalg import eigh


class STO(object):
    """
    We use Rerdberg atomic units.
    """
    def __init__(self, atomic_charge, alpha):
        self.atomic_charge = atomic_charge
        self.alpha = np.array(alpha)
        self.n_alpha = len(self.alpha)
        if self.n_alpha > 100:
            raise ValueError(f"too many gaussians: n_alpha={self.n_alpha}")

        self.qe2 = 2.  # e^2 / (4 * pi * epsilon) = 2 in Rerdberg atomic units!

    def solve(self):
        raise NotImplementedError

    def eigenstate(self, coeffs):
        raise NotImplementedError


class SWave(STO):
    """
    S-wave
    We use Rerdberg atomic units.
    """
    def solve(self):
        alpha_i = self.alpha[:, np.newaxis]
        alpha_j = self.alpha[np.newaxis, :]

        overlap = (np.pi / (alpha_i + alpha_j)) ** 1.5
        H = 6. * alpha_i * alpha_j / (alpha_i + alpha_j) * overlap \
            - 2. * np.pi * self.atomic_charge * self.qe2 / (alpha_i + alpha_j)

        # solve generalized eigenvalue problem
        # i-th eigenvectors is eigvecs[:, i] with eigvals[i]
        eigvals, eigvecs = eigh(H, overlap)
        return eigvals, eigvecs

    def eigenstate(self, coeffs):
        dr = 0.1 / self.atomic_charge
        nrx = 200
        r = np.array(range(nrx)) * dr
        f = np.sum(coeffs[np.newaxis, :] * np.exp(-self.alpha[np.newaxis, :] * r[:, np.newaxis] * r[:, np.newaxis]), axis=1)
        assert(f.shape == r.shape)
        return r, f


class PWave(STO):
    """
    P-wave
    We use Rerdberg atomic units.
    """
    def solve(self):
        alpha_i = self.alpha[:, np.newaxis]
        alpha_j = self.alpha[np.newaxis, :]

        overlap = 0.5 / (alpha_i + alpha_j) * (np.pi / (alpha_i + alpha_j)) ** 1.5
        H = 10. * alpha_i * alpha_j / (alpha_i + alpha_j) * overlap \
            - 2. * np.pi * self.atomic_charge * self.qe2 / (3. * (alpha_i + alpha_j) ** 2)

        # solve generalized eigenvalue problem
        # i-th eigenvectors is eigvecs[:, i] with eigvals[i]
        eigvals, eigvecs = eigh(H, overlap)
        return eigvals, eigvecs

    def eigenstate(self, coeffs):
        dr = 0.1 / self.atomic_charge
        nrx = 200
        r = np.array(range(nrx)) * dr
        f = np.sum(r[:, np.newaxis] * coeffs[np.newaxis, :] \
                   * np.exp(-self.alpha[np.newaxis, :] * r[:, np.newaxis] * r[:, np.newaxis]),
                   axis=1)
        assert(f.shape == r.shape)
        return r, f


if __name__ == '__main__':
    sto_3G = [0.109818, 0.405771, 2.2776]
    sto_4G = [0.121949, 0.444529, 1.962079, 13.00773]
    swave = SWave(1, sto_3G)
    eigvals, eigvecs = swave.solve()
    print(f"{eigvals[0]:.6f} Ry")

    sto_2G_2p = [0.0974545, 0.384244]
    pwave = PWave(1, sto_2G_2p)
    eigvals, eigvecs = pwave.solve()
    print(eigvals)
    print(f"{eigvals[0]:.6f} Ry")
