import numpy as np
import matplotlib.pyplot as plt


class PeriodicWell:
    """
    solve Kronig-Penny model with plane-wave expansion
    V(x) = sum_{n} v(x - n * period)
    v(x) = (x in [-width/2, width/2]) ? -V0 : 0
    """
    def __init__(self, width, V0, period, encut):
        self.width = width
        self.V0 = V0
        self.period = period
        self.encut = encut
        assert(V0 > 0)
        assert(period > 0 and width > 0 and period > width)
        assert(encut > 0)

        # number of plane waves
        npw = np.ceil(np.sqrt(self.encut / (2 * np.pi / self.period) ** 2))
        self.npw = 2 * int(npw) + 1
        print(f" number of plane wave basis: {self.npw}")

        # reciprocal lattice vector
        # g[n] = (n % 2 == 1) ? 2 * pi * n / period : -2 * pi * n / period
        self.g = np.zeros(self.npw)
        self.g[1::2] = np.arange(1, npw + 1) * np.pi / self.period
        self.g[2::2] = -self.g[1::2]

        # compute V(G) with FFT
        self.nfft = 4 * self.npw
        v_real = np.zeros(self.nfft)
        xi = np.arange(self.nfft) * self.period / self.nfft  # in [0, period)
        v_real[xi <= self.width / 2] = -self.V0
        v_real[xi > (self.period - self.width / 2)] = -self.V0

        self.v_real_fft = np.fft.fft(v_real) / self.nfft
        """
        v_real_fft = np.fft.rfft(v_real) / self.nfft
        self.v_real_fft = np.zeros(self.nfft)
        self.v_real_fft[0] = v_real_fft[0]
        self.v_real_fft[1::2] = v_real_fft[1:]
        self.v_real_fft[2::2] = np.conj(v_real_fft[1:])
        """

    def solve(self, kpoint):
        H = np.zeros((self.npw, self.npw))
        for i in range(self.npw):
            for j in range(self.npw):
                if i == j:
                    H[i, j] = (kpoint + self.g[i]) ** 2 + self.v_real_fft[0]
                else:
                    ifft = int(np.around((self.g[i] - self.g[j]) * self.period / (2 * np.pi)))
                    if ifft < 0:
                        ifft += self.nfft
                    H[i, j] = self.v_real_fft[ifft]

        eigvals, eigvecs = np.linalg.eigh(H)
        print(f"lowest energies: {eigvals[0]}, {eigvals[1]}, {eigvals[2]}")
        return eigvals


if __name__ == '__main__':
    width = 0.5
    V0 = 2
    period = 1.
    pw = PeriodicWell(width=width, V0=V0, period=period, encut=100.)

    n = 20
    for m in range(-n, n + 1):
        kpoint = m * np.pi / (n * period)
        eigvals = pw.solve(kpoint)
        plt.scatter([kpoint] * len(eigvals), eigvals, c='b')

    plt.show()
