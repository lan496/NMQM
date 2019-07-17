import numpy as np
from scipy.special import comb
from scipy.linalg import eigh_tridiagonal, eigh


class Heisenberg:
    """
    Exact diagonalization of isotropic 1d Heisenberg model

    Energy in units of J.
    """
    def __init__(self, N, nup, Jsign=1):
        self.Jsign = Jsign
        self.N = N  # number of sites
        self.nup = nup  # number of up-spin site
        assert(0 < self.N < 32)

        # dimension of the Hilbert space
        self.nhil = int(comb(self.N, self.nup))
        # neighbor[j] is the index of the neighbor of the j-th spin
        self.neighbor = (np.arange(self.N) + 1) % self.N

        # represent spin by bit, up-spin iff 1
        self.states = [i for i in range(2 ** self.N) if bin(i).count('1') == self.nup]
        self.mapping = {st: i for i, st in enumerate(self.states)}

        # Hamiltonian matrix
        self.H = np.zeros((self.nhil, self.nhil))
        for st in self.states:
            for i, j in zip(range(self.N), self.neighbor):
                si = (st >> i) & 1  # i-th spin
                sj = (st >> j) & 1  # j-th spin
                idx_st = self.mapping[st]

                # S+(i) S-(j) term
                if (si == 0) and (sj == 1):
                    st_ij = st + (1 << i) - (1 << j)
                    assert(st_ij in self.mapping)
                    self.H[self.mapping[st_ij], idx_st] -= 0.5 * self.Jsign
                # S-(i) S+(j) term
                if (si == 1) and (sj == 0):
                    st_ij = st - (1 << i) + (1 << j)
                    assert(st_ij in self.mapping)
                    self.H[self.mapping[st_ij], idx_st] -= 0.5 * self.Jsign
                # Sz(i) Sz(j) term
                self.H[idx_st, idx_st] -= 0.25 * self.Jsign * (2 * si - 1) * (2 * sj - 1)

    def lanczos(self, max_num_steps, random_state=None):
        assert(self.nhil >= 2)
        num_steps = min(max_num_steps, self.nhil)
        np.random.seed(random_state)

        d = np.zeros(num_steps)  # diagonal part of tridiagonal matrix
        e = np.zeros(num_steps)  # off-diagonal part of tridigaonal matrix
        e[0] = 1

        v0 = np.zeros(self.nhil)
        v1 = np.random.rand(self.nhil)
        v1 /= np.linalg.norm(v1)

        for j in range(2, num_steps + 1):
            d[j - 2] = np.dot(v1, np.dot(self.H, v1))
            wj = np.dot(self.H, v1) - d[j - 2] * v1 - e[j - 2] * v0
            e[j - 1] = np.linalg.norm(wj)

            v0 = v1
            v1 = wj / e[j - 1]

        # import pdb; pdb.set_trace()
        eigvals, eigvecs = eigh_tridiagonal(d, e[1:])
        return eigvals

    def conventional(self):
        eigvals, _ = eigh(self.H)
        return eigvals


def ferro(N):
    nup = N
    hs = Heisenberg(N, nup, Jsign=1)
    energy = hs.conventional()
    assert(np.isclose(energy[0], -N / 4))

    nup = N - 1
    hs = Heisenberg(N, nup, Jsign=1)
    energy_actual = hs.conventional()
    energy_expect = sorted(-N / 4 + 1 - np.cos(2 * np.pi * np.arange(N) / N))
    print(energy_expect)
    assert(np.allclose(energy_actual, energy_expect))
    energy_lanzcos = hs.lanczos(16, random_state=0)
    print(energy_lanzcos)
    assert(np.isclose(energy_lanzcos[0], energy_expect[0]))


if __name__ == '__main__':
    ferro(N=16)
