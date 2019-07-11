import numpy as np
from scipy.linalg import eigh


class HeliumHF:
    """
    Hartree-Fock ground state solution for He atom
    We use Rerdberg atomic units
    """
    def __init__(self, atomic_charge, alpha):
        self.atomic_charge = atomic_charge
        self.alpha = np.array(alpha)
        self.n_alpha = len(self.alpha)
        assert(self.n_alpha >= 1)

        self.qe2 = 2.  # e^2 / (4 * pi * epsilon) = 2 in Rerdberg atomic units!
        self.maxiter = 100
        self.ediff = 1e-8

        # el-el interaction matrix for S-wave
        # ee[i, j, k, l] = < i, j | 1/|r - r'| | k, l >
        alpha_i = self.alpha[:, None, None, None]
        alpha_j = self.alpha[None, :, None, None]
        alpha_k = self.alpha[None, None, :, None]
        alpha_l = self.alpha[None, None, None, :]
        alpha_ik = alpha_i + alpha_k
        alpha_jl = alpha_j + alpha_l
        self.ee = 2 * self.qe2 * (np.pi ** 2.5) / (alpha_ik * alpha_jl * np.sqrt(alpha_ik + alpha_jl))

        # overlap matrix for S-wave
        alpha_p = self.alpha[:, None]
        alpha_q = self.alpha[None, :]
        self.overlap = (np.pi / (alpha_p + alpha_q)) ** 1.5
        self.H = 6. * alpha_p * alpha_q / (alpha_p + alpha_q) * self.overlap \
            - 2. * np.pi * self.atomic_charge * self.qe2 / (alpha_p + alpha_q)

    def solve(self):
        # coefficients for basis functions
        # coeff[:, k] is for k-th basis function
        coeff = np.zeros((self.n_alpha, self.n_alpha))
        coeff[0, 0] = 1 / np.sqrt(self.overlap[0, 0] ** 2)
        density = self.get_density(coeff)

        eold = 0.
        for i in range(self.maxiter):
            fock = self.H + self.couloumb_exchange(density)

            # solve generalized eigenvalue problem
            # i-th eigenvectors is eigvecs[:, i] with eigvals[i]
            eigvals, eigvecs = eigh(fock, self.overlap)

            # normalize coefficients
            coeff = eigvecs
            coeff[:, 1:] = 0  # occupy only ground state
            coeff[:, 0] /= np.sqrt(np.dot(eigvecs[:, 0].T, np.dot(self.overlap, eigvecs[:, 0])))
            density = self.get_density(coeff)

            enew = np.sum(density * (self.H + 0.5 * self.couloumb_exchange(density)))
            print(f" Iteration # {i:2d}: HF eigenvalue, energy: {eigvals[0]} {enew}")

            if np.abs(enew - eold) < self.ediff:
                print(" Convergence archived, stopping")
                return enew, eigvals, eigvecs

            eold = enew

        print(" Convergence not reached, stopping")
        return None

    def get_density(self, coeff):
        return 2. * np.dot(coeff, coeff.T)

    def couloumb_exchange(self, density):
        g = np.einsum("rs,prqs->pq", density, self.ee) \
            - 0.5 * np.einsum("rs,prsq->pq", density, self.ee)
        return g


if __name__ == '__main__':
    sto_3G = [0.109818, 0.405771, 2.2776]
    alpha = [0.297104, 1.236745, 5.749982, 38.2166677]

    hf = HeliumHF(2, alpha)
    hf.solve()
