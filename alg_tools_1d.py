from __future__ import division
import numpy as np
from scipy import linalg
import os
from matplotlib import rcParams

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True


def distance(x1, x2):
    """
    Given two arrays of numbers x1 and x2, pairs the cells that are the
    closest and provides the pairing matrix index: x1(index(1,:)) should be as
    close as possible to x2(index(2,:)). The function outputs the average of the
    absolute value of the differences abs(x1(index(1,:))-x2(index(2,:))).
    :param x1: vector 1
    :param x2: vector 2
    :return: d: minimum distance between d
             index: the permutation matrix
    """
    x1 = np.reshape(x1, (1, -1), order='F')
    x2 = np.reshape(x2, (1, -1), order='F')
    N1 = x1.size
    N2 = x2.size
    diffmat = np.abs(x1 - np.reshape(x2, (-1, 1), order='F'))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in range(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float('inf')
            diffmat[:, index1] = float('inf')
        d = np.mean(np.abs(x1[:, index[:, 0]] - x2[:, index[:, 1]]))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index


def periodicSinc(t, M):
    numerator = np.sin(t)
    denominator = M * np.sin(t / M)
    idx = np.abs(denominator) < 1e-12
    numerator[idx] = np.cos(t[idx])
    denominator[idx] = np.cos(t[idx] / M)
    return numerator / denominator


def Tmtx(data, K):
    """Construct convolution matrix for a filter specified by 'data'
    """
    return linalg.toeplitz(data[K::], data[K::-1])


def Rmtx(data, K, seq_len):
    """A dual convolution matrix of Tmtx. Use the commutativness of a convolution:
    a * b = b * c
    Here seq_len is the INPUT sequence length
    """
    col = np.concatenate(([data[-1]], np.zeros(seq_len - K - 1)))
    row = np.concatenate((data[::-1], np.zeros(seq_len - K - 1)))
    return linalg.toeplitz(col, row)


def tri(x):
    """triangular interpolation kernel:
    tri(x) = 1 - x  if x in [0,1]
           = 1 + x  if x in [-1,0)
           = 0      otherwise
    """
    y = np.zeros(x.shape)
    idx1 = np.bitwise_and(x >= 0, x <= 1)
    idx2 = np.bitwise_and(x >= -1, x < 0)
    y[idx1] = 1 - x[idx1]
    y[idx2] = 1 + x[idx2]
    return y


def cubicSpline(x):
    """cubic spline interpolation kernel
    """
    y = np.zeros(x.shape)
    idx1 = np.bitwise_and(x >= -2, x < -1)
    idx2 = np.bitwise_and(x >= -1, x < 0)
    idx3 = np.bitwise_and(x >= 0, x < 1)
    idx4 = np.bitwise_and(x >= 1, x < 2)
    y[idx1] = (x[idx1] + 2) ** 3 / 6.
    y[idx2] = -0.5 * x[idx2] ** 3 - x[idx2] ** 2 + 2 / 3.
    y[idx3] = 0.5 * x[idx3] ** 3 - x[idx3] ** 2 + 2 / 3.
    y[idx4] = -(y[idx4] - 2) ** 3 / 6.
    return y


def keysInter(x):
    """Keys interpolation function
    """
    y = np.zeros(x.shape)
    abs_x = np.abs(x)
    idx1 = np.bitwise_and(abs_x >= 0, abs_x < 1)
    idx2 = np.bitwise_and(abs_x >= 1, abs_x <= 2)
    y[idx1] = 1.5 * abs_x[idx1] ** 3 - 2.5 * abs_x[idx1] ** 2 + 1
    y[idx2] = -0.5 * abs_x[idx2] ** 3 + 2.5 * abs_x[idx2] ** 2 - 4 * abs_x[idx2] + 2
    return y


def build_G_fourier(omega_ell, M, tau, interp_kernel, **kwargs):
    """
    build a linear mapping matrix that links the Fourier transform on a uniform grid
    to the given Fourier domain measurements by using the current reconstructed Dirac locations
    :param omega_ell: the frequency where the Fourier transforms are measured
    :param M: the spectrum between -M*pi and M*pi is considered
    :param tau: time support of the Diracs are between -0.5*tau to 0.5*tau
    :param interp_kernel: interpolation kernel assumed
    :param tk_ref: reference locations of the Dirac, e.g., from previous reconstruction
    :return:
    """
    m_limit = (np.floor(M * tau / 2.)).astype(int)
    m_grid, omegas = np.meshgrid(np.arange(-m_limit, m_limit + 1), omega_ell)
    if interp_kernel == 'dirichlet':
        Phi_inter = periodicSinc((tau * omegas - 2 * np.pi * m_grid) / 2., M * tau)
    elif interp_kernel == 'triangular':
        Phi_inter = tri(omegas / (2 * np.pi / tau) - m_grid)
    elif interp_kernel == 'cubic':
        Phi_inter = cubicSpline(omegas / (2 * np.pi / tau) - m_grid)
    elif interp_kernel == 'keys':
        Phi_inter = keysInter(omegas / (2 * np.pi / tau) - m_grid)
    else:
        Phi_inter = periodicSinc((tau * omegas - 2 * np.pi * m_grid) / 2., M * tau)
    if 'tk_ref' in kwargs:
        tk_ref = kwargs['tk_ref']
        # now the part that is build based on tk_ref
        tks_grid, m_uni_grid = np.meshgrid(tk_ref, np.arange(-m_limit, m_limit + 1))
        B_ref = np.exp(-1j * 2 * np.pi / tau * m_uni_grid * tks_grid)
        W_ref = linalg.solve(np.dot(B_ref.conj().T, B_ref), B_ref.conj().T)
        # the orthogonal complement
        W_ref_orth = np.eye(2 * m_limit + 1) - np.dot(B_ref, W_ref)
        tks_grid, omega_ell_grid = np.meshgrid(tk_ref, omega_ell)
        G = np.dot(Phi_inter, W_ref_orth) + \
            np.dot(np.exp(-1j * omega_ell_grid * tks_grid), W_ref)
    else:
        G = Phi_inter
    return G


def dirac_recon_time(G, a, K, noise_level, max_ini=100, stop_cri='mse'):
    compute_mse = (stop_cri == 'mse')
    M = G.shape[1]
    GtG = np.dot(G.conj().T, G)
    Gt_a = np.dot(G.conj().T, a)

    max_iter = 50
    min_error = float('inf')
    # beta = linalg.solve(GtG, Gt_a)
    beta = linalg.lstsq(G, a)[0]

    Tbeta = Tmtx(beta, K)
    rhs = np.concatenate((np.zeros(2 * M + 1), [1.]))
    rhs_bl = np.concatenate((Gt_a, np.zeros(M - K)))

    for ini in range(max_ini):
        c = np.random.randn(K + 1) + 1j * np.random.randn(K + 1)
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = Rmtx(c, K, M)

        # first row of mtx_loop
        mtx_loop_first_row = np.hstack((np.zeros((K + 1, K + 1)), Tbeta.conj().T,
                                        np.zeros((K + 1, M)), c0[:, np.newaxis]))
        # last row of mtx_loop
        mtx_loop_last_row = np.hstack((c0[np.newaxis].conj(),
                                       np.zeros((1, 2 * M - K + 1))))

        for loop in range(max_iter):
            mtx_loop = np.vstack((mtx_loop_first_row,
                                  np.hstack((Tbeta, np.zeros((M - K, M - K)),
                                             -R_loop, np.zeros((M - K, 1)))),
                                  np.hstack((np.zeros((M, K + 1)), -R_loop.conj().T,
                                             GtG, np.zeros((M, 1)))),
                                  mtx_loop_last_row
                                  ))

            # matrix should be Hermitian symmetric
            mtx_loop += mtx_loop.conj().T
            mtx_loop *= 0.5
            # mtx_loop = (mtx_loop + mtx_loop.conj().T) / 2.

            c = linalg.solve(mtx_loop, rhs)[:K + 1]

            R_loop = Rmtx(c, K, M)

            mtx_brecon = np.vstack((np.hstack((GtG, R_loop.conj().T)),
                                    np.hstack((R_loop, np.zeros((M - K, M - K))))
                                    ))

            # matrix should be Hermitian symmetric
            mtx_brecon += mtx_brecon.conj().T
            mtx_brecon *= 0.5
            # mtx_brecon = (mtx_brecon + mtx_brecon.conj().T) / 2.

            b_recon = linalg.solve(mtx_brecon, rhs_bl)[:M]

            error_seq[loop] = linalg.norm(a - np.dot(G, b_recon))
            if error_seq[loop] < min_error:
                min_error = error_seq[loop]
                b_opt = b_recon
                c_opt = c
            if min_error < noise_level and compute_mse:
                break
        if min_error < noise_level and compute_mse:
            break

    return b_opt, min_error, c_opt, ini


def dirac_recon_irreg_fourier(FourierData, K, tau, omega_ell, M, noise_level,
                              max_ini=100, stop_cri='mse', interp_kernel='dirichlet',
                              update_G=False):
    # whether to update the linear transformation matrix G
    # based on previous reconstructions or not
    error_opt = float('inf')

    if update_G:
        max_outer = 50
    else:
        max_outer = 1

    for outer in range(max_outer):
        if outer == 0:
            G = build_G_fourier(omega_ell, M, tau, interp_kernel)
        else:
            # use the previous reconstruction to build new linear mapping matrix Phi
            G = build_G_fourier(omega_ell, M, tau, interp_kernel, tk_ref=tk_opt)

        # FRI reconstruction
        b_recon, min_error, c_opt = \
            dirac_recon_time(G, FourierData, K, noise_level, max_ini, stop_cri)[:3]

        if outer == 0:
            print(r'Noise level: {0:.2e}'.format(noise_level))

        # reconstruct Diracs' locations tk
        z = np.roots(c_opt)
        z = z / np.abs(z)
        tk_recon = np.real(tau * 1j / (2 * np.pi) * np.log(z))
        # round to [-tau/2,tau/2]
        tk_recon = np.sort(tk_recon - np.floor((tk_recon + 0.5 * tau) / tau) * tau)

        if min_error < error_opt:
            error_opt = min_error
            tk_opt = tk_recon
            b_opt = b_recon

        if error_opt < noise_level:
            break

    print(r'Minimum approximation error |a - Gb|_2: {0:.2e}'.format(error_opt))
    # reconstruct amplitudes ak
    tk_recon_grid, freq_grid = np.meshgrid(tk_opt, omega_ell)
    Phi_amp = np.exp(-1j * freq_grid * tk_recon_grid)
    alphak_recon = np.real(linalg.solve(np.dot(Phi_amp.conj().T, Phi_amp),
                                        np.dot(Phi_amp.conj().T, FourierData)))
    return tk_opt, alphak_recon, b_opt
