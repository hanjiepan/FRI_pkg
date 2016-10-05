from __future__ import division
import numpy as np
import scipy as sp
from scipy import linalg
import scipy.optimize
import scipy.sparse as sps
import scipy.fftpack as sp_fft
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from functools import partial
from joblib import Parallel, delayed, cpu_count
import os
from matplotlib import rcParams
import matplotlib.pyplot as plt
from alg_tools_1d import distance

use_mkl_fft = True
try:
    import mkl_fft
except ImportError:
    use_mkl_fft = False

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + \
                     ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True


def distance_2d(x, y, x_ref, y_ref):
    z = x + 1j * y
    z_ref = x_ref + 1j * y_ref
    ind = distance(z, z_ref)[1]
    return np.sqrt(np.mean(np.abs(z[ind[:, 0]] - z_ref[ind[:, 1]]) ** 2))


def snr_normalised(data, data_ref, rescale=True):
    """
    Compute a scale invariant SNR: SNR(alpha * data, data_ref).
    Here alpha is a scalar that is computed by minimising the difference
    between alpha * data and data_ref.
    :param data: input data
    :param data_ref: reference data
    :param rescale: wether compute the scaling factor alpha or not. If not, then alpha = 1
    :return:
    """
    data = np.reshape(data, (-1, 1), order='F')
    data_ref = np.reshape(data_ref, (-1, 1), order='F')
    if rescale:
        alpha = np.dot(data.conj().T, data_ref).squeeze() / np.dot(data.conj().T, data).squeeze()
    else:
        alpha = 1.
    return 20 * np.log10(linalg.norm(data_ref) / linalg.norm(alpha * data - data_ref))


def std_normalised(data, data_ref):
    """
    Compute a normalised standard deviation for the difference
    between the data and the reference data_ref:
        std(alpha * data - data_ref)
    Here alpha is a scalar that is computed by minimising the
    difference between alpha * data and data_ref.
    :param data: input data
    :param data_ref: reference data
    :return:
    """
    data = np.reshape(data, (-1, 1), order='F')
    data_ref = np.reshape(data_ref, (-1, 1), order='F')
    alpha = np.dot(data.conj().T, data_ref).squeeze() / np.dot(data.conj().T, data).squeeze()
    return np.std(alpha * data - data_ref), alpha


def roots2(coef, plt_sz, tau_x, tau_y):
    """
    Find the roots of a 2D polynomial with the coefficients specified by c_{k,l}
    by utilizing the np.roots function along x/y-directions seperately. The mask
    function is given as

        mu(x,y) = sum_{k,l} c_{k,l} e^{2j*pi*k/tau_x * x + 2j*pi*l/tau_y * y}

    :param coef: curve coefficients
    :param plt_sz: size of the plot. plt_sz[0] corresponds to the vertical dimension, i.e., y-axis
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :return:
    """
    L, K = coef.shape
    l_limit = np.int(np.floor(L / 2.))
    k_limit = np.int(np.floor(K / 2.))

    x_grid = np.linspace(0, tau_x, num=plt_sz[1], endpoint=False, dtype=float)
    y_grid = np.linspace(0, tau_y, num=plt_sz[0], endpoint=False, dtype=float)
    # evaluate the vertical direction roots
    vec_K = np.arange(-k_limit, k_limit + 1, dtype=float)[:, np.newaxis]
    j2pi_taux = 1j * 2 * np.pi / tau_x
    j2pi_tauy = 1j * 2 * np.pi / tau_y
    cords = np.empty((0, 2))
    for loop in range(np.int(plt_sz[1])):
        x_loop = x_grid[loop]
        root_loop = np.roots((np.dot(coef,
                                     np.exp(j2pi_taux * x_loop * vec_K)
                                     ))[::-1].squeeze())
        pos_loop = np.log(root_loop) / j2pi_tauy
        idx = np.abs(np.imag(pos_loop)) < 1e-10
        y_cord = np.real(pos_loop[idx])
        y_cord -= np.floor(y_cord / tau_y) * tau_y
        cords = np.concatenate((cords, np.hstack((np.tile(x_loop, (y_cord.size, 1)),
                                                  y_cord[:, np.newaxis]
                                                  ))
                                ), axis=0)

    # evaluate the horizontal direction roots
    vec_L = np.arange(-l_limit, l_limit + 1, dtype=float)[:, np.newaxis]
    coef_T = coef.T
    for loop in range(np.int(plt_sz[0])):
        y_loop = y_grid[loop]
        root_loop = np.roots((np.dot(coef_T,
                                     np.exp(j2pi_tauy * y_loop * vec_L)
                                     ))[::-1].squeeze())
        pos_loop = np.log(root_loop) / j2pi_taux
        idx = np.abs(np.imag(pos_loop)) < 1e-10
        x_cord = np.real((pos_loop[idx]))
        x_cord -= np.floor(x_cord / tau_x) * tau_x
        cords = np.concatenate((cords, np.hstack((x_cord[:, np.newaxis],
                                                  np.tile(y_loop, (x_cord.size, 1))
                                                  ))
                                ), axis=0)

    # normalize to the closet discrete grids speicified by [x_grid,y_grid]
    Tx = tau_x / plt_sz[1]
    Ty = tau_y / plt_sz[0]
    #    eps = np.spacing(1)
    cords_normalise = np.hstack((np.floor(cords[:, 0] / Tx)[:, np.newaxis],
                                 np.floor(cords[:, 1] / Ty)[:, np.newaxis]
                                 ))
    curve_plt = np.zeros(np.int(plt_sz[0] * plt_sz[1]))
    curve_plt[(cords_normalise[:, 0] * plt_sz[0] +
               cords_normalise[:, 1]).astype(int)] = 1
    curve_plt = np.reshape(curve_plt, tuple(plt_sz.astype(int)), order='F')
    return curve_plt


def convmtx2(H, M, N):
    """
    build 2d convolution matrix
    :param H: 2d filter
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :return:
    """
    P, Q = H.shape
    blockHeight = int(M + P - 1)
    blockWidth = int(M)
    blockNonZeros = int(P * M)
    totalNonZeros = int(Q * N * blockNonZeros)

    THeight = int((N + Q - 1) * blockHeight)
    TWidth = int(N * blockWidth)

    Tvals = np.zeros((totalNonZeros, 1), dtype=H.dtype)
    Trows = np.zeros((totalNonZeros, 1))
    Tcols = np.zeros((totalNonZeros, 1))

    c = np.dot(np.diag(np.arange(1, M + 1)), np.ones((M, P), dtype=float))
    r = np.repeat(np.reshape(c + np.arange(0, P)[np.newaxis], (-1, 1), order='F'), N, axis=1)
    c = np.repeat(c.flatten('F')[:, np.newaxis], N, axis=1)

    colOffsets = np.arange(N) * M
    colOffsets = np.reshape(np.repeat(colOffsets[np.newaxis], M * P, axis=0) + c, (-1, 1), order='F') - 1

    rowOffsets = np.arange(N) * blockHeight
    rowOffsets = np.reshape(np.repeat(rowOffsets[np.newaxis], M * P, axis=0) + r, (-1, 1), order='F') - 1

    for k in range(Q):
        val = np.reshape(np.tile((H[:, k]).flatten(), (M, 1)), (-1, 1), order='F')
        first = int(k * N * blockNonZeros)
        last = int(first + N * blockNonZeros)
        Trows[first:last] = rowOffsets
        Tcols[first:last] = colOffsets
        Tvals[first:last] = np.tile(val, (N, 1))
        rowOffsets += blockHeight

    T = sps.coo_matrix((Tvals.squeeze(), (Trows.squeeze(), Tcols.squeeze())),
                       shape=(THeight, TWidth)).toarray()
    return T


def convmtx2_valid(H, M, N):
    """
    2d convolution matrix with the boundary condition 'valid', i.e., only filter
    within the given data block.
    :param H: 2d filter
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :return:
    """
    T = convmtx2(H, M, N)
    s_H0, s_H1 = H.shape
    if M >= s_H0:
        S = np.pad(np.ones((M - s_H0 + 1, N - s_H1 + 1), dtype=bool),
                   ((s_H0 - 1, s_H0 - 1), (s_H1 - 1, s_H1 - 1)),
                   'constant', constant_values=False)
    else:
        S = np.pad(np.ones((s_H0 - M + 1, s_H1 - N + 1), dtype=bool),
                   ((M - 1, M - 1), (N - 1, N - 1)),
                   'constant', constant_values=False)
    T = T[S.flatten('F'), :]
    return T


def create_fri_curve_mask(coef, len_x, len_y, tau_x, tau_y):
    """
    Crete a mask function for FRI curve from the given curve coefficients:
        mu(x,y) = sum_{k,l} c_{k,l} exp(jkx/tau_x + jly/tau_y)
    The FRI curve is defined as the roots of the mask function: mu(x,y) = 0.
    :param coef: a 2D array curve coefficients
    :param len_x: length of the horizontal axis for the mask function
    :param len_y: length of the vertical axis for the mask function
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :return:
    """
    # zero padding the coefficients to the same size of x and y
    coef_pad = np.pad(coef, ((0, (len_y - coef.shape[0]).astype(int)),
                             (0, (len_x - coef.shape[1]).astype(int))),
                      mode='constant', constant_values=0)
    # circular shift the origin to the upper-left corner
    coef_shift = np.roll(np.roll(coef_pad, -np.int(np.floor(coef.shape[0] / 2.)), 0),
                         -np.int(np.floor(coef.shape[1] / 2.)), 1)
    if use_mkl_fft:
        mask_fn = mkl_fft.ifft2(coef_shift) * len_x * len_y / (tau_x * tau_y)
    else:
        mask_fn = sp_fft.fft2(coef_shift) * len_x * len_y / (tau_x * tau_y)
    return mask_fn


def gen_samples_edge_img(coef, samp_size, B_x, B_y, tau_x, tau_y, amp_fn=lambda x, y: 1., over_samp_ratio=221):
    """
    Generate the ideally lowpass filtered samples (with fft2) of an edge image,
    which is discontinuous on the curve with the curve coefficients coef.
    :param coef: curve coefficients that define the FRI curve
    :param samp_size: 1 x 2 array, desired sample size
    :param B_x: bandwidth of the ideal lowpass filter along x-axis
    :param B_y: bandwidth of the ideal lowpass filter along y-axis
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :param amp_fn: the amplitude function. The default is 1, i.e.,
            the edge image is the indicator function
    :param over_samp_ratio: the over sampling ratio used in fft. The higher the ratio,
            the accurater the fft approximation is.
    :return:
    """
    len_x = samp_size[1] * over_samp_ratio
    len_y = samp_size[0] * over_samp_ratio
    xs = np.reshape(np.linspace(0, tau_x, len_x, endpoint=False), (1, -1), order='F')
    ys = np.reshape(np.linspace(0, tau_y, len_y, endpoint=False), (-1, 1), order='F')
    # number of frequency domain samples
    freq_samp_sz_x = np.int(2 * np.floor(B_x * tau_x / 2) + 1)
    freq_samp_sz_y = np.int(2 * np.floor(B_y * tau_y / 2) + 1)
    # boundary width (assume that both samp_size and len_x are ODD numbers!!!
    assert samp_size[0] % 2 == 1 and samp_size[1] % 2 == 1
    wd0 = np.int((len_y - freq_samp_sz_y) / 2.)
    wd1 = np.int((len_x - freq_samp_sz_x) / 2.)

    mask_fn = np.real(create_fri_curve_mask(coef, len_x, len_y, tau_x, tau_y))
    edge_img = np.double(mask_fn <= 0.) * amp_fn(xs, ys)

    # ideal lowpass filtering and sampling (use fft2 for numerical implementation)
    if use_mkl_fft:
        fourier_trans_all = sp_fft.fftshift(mkl_fft.fft2(edge_img))
    else:
        fourier_trans_all = sp_fft.fftshift(sp_fft.fft2(edge_img))

    # extracting the low frequency components based on the bandwidth
    fourier_lowpass = fourier_trans_all[wd0:wd0 + freq_samp_sz_y,
                      wd1:wd1 + freq_samp_sz_x] / ((len_x * len_y) /
                                                   (freq_samp_sz_x * freq_samp_sz_y))

    # pad zeros based on the size of the spatial domain samples
    pad_sz_x = np.int((samp_size[1] - freq_samp_sz_x) / 2)
    pad_sz_y = np.int((samp_size[0] - freq_samp_sz_y) / 2)
    fourier_data = np.pad(fourier_lowpass,
                          ((pad_sz_y, pad_sz_y), (pad_sz_x, pad_sz_x)),
                          mode='constant', constant_values=0)
    if use_mkl_fft:
        samples = mkl_fft.ifft2(sp_fft.ifftshift(fourier_data)) * \
                  (samp_size[0] * samp_size[1] / (freq_samp_sz_x * freq_samp_sz_y))
    else:
        samples = sp_fft.ifft2(sp_fft.ifftshift(fourier_data)) * \
                  (samp_size[0] * samp_size[1] / (freq_samp_sz_x * freq_samp_sz_y))
    return samples, fourier_lowpass


def build_G_fri_curve(x_samp, y_samp, B_x, B_y, tau_x, tau_y):
    """
    :param x_samp: sampling locations along x-axis
    :param y_samp: sampling locations along y-axis
    :param B_x: filter bandwidth along x-axis
    :param B_y: filter bandwidth along y-axis
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :return:
    """
    k_limit = np.floor(B_x * tau_x / 2.)
    l_limit = np.floor(B_y * tau_y / 2.)
    k_grid, l_grid = np.meshgrid(np.arange(-k_limit, k_limit + 1),
                                 np.arange(-l_limit, l_limit + 1))
    x_grid, y_grid = np.meshgrid(x_samp, y_samp)
    k_grid = np.reshape(k_grid, (1, -1), order='F')
    l_grid = np.reshape(l_grid, (1, -1), order='F')
    x_grid = np.reshape(x_grid, (-1, 1), order='F')
    y_grid = np.reshape(y_grid, (-1, 1), order='F')
    G = np.exp(1j * 2 * np.pi / tau_x * x_grid * k_grid +
               1j * 2 * np.pi / tau_y * y_grid * l_grid) / (B_x * B_y * tau_x * tau_y)
    return G


def Tmtx_curve(data, M, N, freq_factor):
    return convmtx2_valid(data * np.reshape(freq_factor, data.shape, order='F'), M, N)


def Rmtx_curve(coef, M, N, freq_factor):
    """
    build the right dual matrix for the FRI curve
    :param coef: annihilating filter coefficients
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :param freq_factor: the scaling factor in the annihilation constraint
    :return:
    """
    return convmtx2_valid(coef, M, N) * np.reshape(freq_factor, (1, -1), order='F')


def hermitian_expansion(len_c):
    """
    create the expansion matrix such that we expand the vector that is Hermitian symmetric.
    The input vector is the concatenation of the real part and imaginary part
    of the vector in the first half.
    :param len_c: length of the first half for the real part. Hence, it is 1 element more than
                  that for the imaginary part
    :return: D1: expansion matrix for the real part
             D2: expansion matrix for the imaginary part
    """
    D0 = np.eye(len_c)
    D1 = np.vstack((D0, D0[1::, ::-1]))
    D2 = np.vstack((D0, -D0[1::, ::-1]))
    D2 = D2[:, :-1]
    return D1, D2


def recon_fri_curve(G, a, K, L, B_x, B_y, tau_x, tau_y, noise_level, max_ini=100, stop_cri='mse'):
    """
    Reconstruction FRI curve from the given set of ideally lowpass filtered samples
    :param G: the linear mapping between the measurements a and the unknown FIR sequence b
    :param a: the given measurements
    :param K: size of the curve coefficients along x-axis (2K_0 + 1)
    :param L: size of the curve coefficients along y-axis (2L_0 + 1)
    :param B_x: bandwidth of the lowpass filter along x-axis
    :param B_y: bandwidth of the lowpass filter along y-axis
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :param noise_level: level of noise in the given measurements
    :param max_ini: maximum number of initialisations
    :param stop_cri: stopping criteria: 1) mse; or 2) max_iter
    :return:
    """
    compute_mse = (stop_cri == 'mse')

    k_limit = np.int(np.floor(B_x * tau_x / 2.))
    l_limit = np.int(np.floor(B_y * tau_y / 2.))

    sz_b0 = 2 * l_limit + 1
    sz_b1 = 2 * k_limit + 1
    sz_conv_out0 = sz_b0 - L + 1
    sz_conv_out1 = sz_b1 - K + 1

    numel_conv_out = sz_conv_out0 * sz_conv_out1
    numel_coef = K * L
    numel_b = sz_b0 * sz_b1

    k_grid, l_grid = np.meshgrid(np.arange(-k_limit, k_limit + 1),
                                 np.arange(-l_limit, l_limit + 1))
    k_grid = np.reshape(k_grid, (-1, 1), order='F')
    l_grid = np.reshape(l_grid, (-1, 1), order='F')
    # the scaling factor in the annihilation constraint
    freq_scaling = 2 * np.pi / tau_x * k_grid + 1j * 2 * np.pi / tau_y * l_grid

    # reshape a as a column vector
    a = np.reshape(a, (-1, 1), order='F')
    GtG = np.dot(G.conj().T, G)
    Gt_a = np.dot(G.conj().T, a)

    max_iter = 50
    min_error = float('inf')

    beta = np.reshape(linalg.lstsq(G, a)[0], (sz_b0, sz_b1), order='F')
    Tbeta = Tmtx_curve(beta, L, K, freq_scaling)
    rhs = np.concatenate((np.zeros(numel_coef + numel_conv_out + numel_b), [1.]))
    rhs_bl = np.concatenate((Gt_a, np.zeros((numel_conv_out, 1), dtype=Gt_a.dtype)))

    for ini in range(max_ini):
        c = np.random.randn(L, K) + 1j * np.random.randn(L, K)
        c0 = np.reshape(c.copy(), (-1, 1), order='F')
        error_seq = np.zeros(max_iter)
        R_loop = Rmtx_curve(c, sz_b0, sz_b1, freq_scaling)

        for loop in range(max_iter):
            # update c
            Mtx_loop = np.vstack((np.hstack((np.zeros((numel_coef, numel_coef)), Tbeta.conj().T,
                                             np.zeros((numel_coef, numel_b)), c0)),
                                  np.hstack((Tbeta, np.zeros((numel_conv_out, numel_conv_out)),
                                             -R_loop, np.zeros((numel_conv_out, 1)))),
                                  np.hstack((np.zeros((numel_b, numel_coef)), -R_loop.conj().T,
                                             GtG, np.zeros((numel_b, 1)))),
                                  np.hstack((c0.conj().T, np.zeros((1, numel_conv_out + numel_b + 1))))
                                  ))
            # matrix should be Hermitian symmetric
            Mtx_loop += Mtx_loop.conj().T
            Mtx_loop *= 0.5
            c = np.reshape(linalg.solve(Mtx_loop, rhs)[:numel_coef],
                           (L, K), order='F')

            # update b
            R_loop = Rmtx_curve(c, sz_b0, sz_b1, freq_scaling)
            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.conj().T)),
                                    np.hstack((R_loop, np.zeros((numel_conv_out, numel_conv_out))))
                                    ))
            # matrix should be Hermitian symmetric
            Mtx_brecon += Mtx_brecon.conj().T
            Mtx_brecon *= 0.5
            b_recon = linalg.solve(Mtx_brecon, rhs_bl)[:numel_b]
            # calculate the objective function value
            error_seq[loop] = linalg.norm(a - np.dot(G, b_recon))
            if error_seq[loop] < min_error:
                min_error = error_seq[loop]
                b_opt = b_recon
                c_opt = c
            # check the stopping criterion
            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    b_opt = np.reshape(b_opt, (sz_b0, sz_b1), order='F')

    # apply least square to the denoised Fourier data
    numel_coef_h = np.int(np.floor(K * L / 2.))

    # take the Hermitian symmetry into account
    D1, D2 = hermitian_expansion(numel_coef_h + 1)
    # D =[D1, 0; 0, D2]
    D = np.vstack((np.hstack((D1, np.zeros((2 * numel_coef_h + 1, numel_coef_h), dtype=float))),
                   np.hstack((np.zeros((2 * numel_coef_h + 1, numel_coef_h + 1), dtype=float), D2))
                   ))
    D1_jD2 = np.hstack((D1, 1j * D2))

    fri_data = b_opt * np.reshape(freq_scaling, (sz_b0, sz_b1), order='F')
    anni_mtx_r = convmtx2_valid(np.real(fri_data), L, K)
    anni_mtx_i = convmtx2_valid(np.imag(fri_data), L, K)
    anni_mtx_ri_D = np.dot(np.vstack((np.hstack((anni_mtx_r, -anni_mtx_i)),
                                      np.hstack((anni_mtx_i, anni_mtx_r))
                                      )), D)

    Vh = linalg.svd(anni_mtx_ri_D, compute_uv=True)[2]
    c_h_ri = Vh.conj().T[:, -1]

    coef = np.reshape(np.dot(D1_jD2, c_h_ri), (L, K), order='F')
    return b_opt, min_error, coef, ini


def plt_recon_fri_curve(coef, coef_ref, tau_x, tau_y, plt_size=(1e3, 1e3),
                        save_figure=False, file_name='fri_curve.pdf',
                        file_format='pdf', nargout=0):
    """
    Plot the reconstructed FRI curve, which is specified by the coefficients.
    :param coef: FRI curve coefficients
    :param coef_ref: ground truth FRI curve coefficients
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :param plt_size: size of the plot to draw the curve as a discrete image
    :param save_figure: save the figure as pdf file or not
    :param file_name: the file name used for saving the figure
    :return:
    """
    curve = 1 - roots2(coef, plt_size, tau_x, tau_y)
    curve_ref = 1 - roots2(coef_ref, plt_size, tau_x, tau_y)
    idx = np.argwhere(curve_ref == 0)
    idx_y = idx[:, 0]
    idx_x = idx[:, 1]
    subset_idx = (np.round(np.linspace(0, idx_x.size - 1, np.int(0.1 * idx_x.size))
                           )).astype(int)

    plt.figure(figsize=(3, 3), dpi=90)
    plt.imshow(255 * curve, origin='upper', vmin=0, vmax=255, cmap='gray')
    plt.scatter(idx_x[subset_idx], idx_y[subset_idx], s=1, edgecolor='none', c=[1, 0, 0], hold=True)
    plt.axis('off')
    if save_figure:
        plt.savefig(file_name, format=file_format, dpi=300, transparent=True)
    # plt.show()
    if nargout == 0:
        return None
    else:
        return curve, idx_x, idx_y, subset_idx


# == for structured low rank approximation (SLRA) method == #
def slra_fri_curve(G, a, K, L, K_alg, L_alg, B_x, B_y, tau_x, tau_y,
                   max_iter=100, weight_choice='1'):
    """
    sturctured low rank approximation method by L. Condat:
    http://www.gipsa-lab.grenoble-inp.fr/~laurent.condat/download/pulses_recovery.m

    :param G: the linear mapping between the spatial domain samples and the Fourier samples.
              Typically a truncated DFT transformation.
    :param a: the spatial domain samples
    :param K: size of the curve coefficients along x-axis (2K_0 + 1)
    :param L: size of the curve coefficients along y-axis (2L_0 + 1)
    :param K_cad: size of the assumed curve coefficients along x-axis,
                  which is used in the Cadzow iterative denoising. It should be at least K.
    :param L_cad: size of the assumed curve coefficients along y-axis,
                  # which is used in the Cadzow iterative denoising. It should be at least L.
    :param B_x: bandwidth of the lowpass filter along x-axis
    :param B_y: bandwidth of the lowpass filter along y-axis
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :param max_iter: maximum number of iterations
    :return:
    """
    mu = 0.1  # default parameter by the author. Must be in (0,2)
    gamma = 0.51 * mu  # default parameter by the author. Must be in (mu/2,1)

    k_limit = np.int(np.floor(B_x * tau_x / 2.))
    l_limit = np.int(np.floor(B_y * tau_y / 2.))
    k_grid, l_grid = np.meshgrid(np.arange(-k_limit, k_limit + 1),
                                 np.arange(-l_limit, l_limit + 1))
    k_grid = np.reshape(k_grid, (-1, 1), order='F')
    l_grid = np.reshape(l_grid, (-1, 1), order='F')

    sz_b0 = 2 * l_limit + 1
    sz_b1 = 2 * k_limit + 1
    b = np.reshape(linalg.lstsq(G, np.reshape(a, (-1, 1), order='F'))[0],
                   (sz_b0, sz_b1), order='F')

    # number of zeros of the annihilating filter matrix in the Cadzow iteration
    num_zero = (K_alg - K + 1) * (L_alg - L + 1)
    blk_sz0 = b.shape[0] - L_alg + 1
    blk_sz1 = L_alg
    num_blk0 = b.shape[1] - K_alg + 1
    num_blk1 = K_alg

    # the scaling factor in the annihilation constraint
    freq_scaling = 2 * np.pi / tau_x * k_grid + 1j * 2 * np.pi / tau_y * l_grid

    # take the Hermtian symmetry into account
    numel_coef_h = np.int(np.floor(K * L / 2))
    D1, D2 = hermitian_expansion(numel_coef_h + 1)

    D = np.vstack((np.hstack((D1, np.zeros((2 * numel_coef_h + 1, numel_coef_h), dtype=float))),
                   np.hstack((np.zeros((2 * numel_coef_h + 1, numel_coef_h + 1), dtype=float), D2))
                   ))

    D1_jD2 = np.hstack((D1, 1j * D2))

    I_hat = b * np.reshape(freq_scaling, b.shape, order='F')
    anni_mtx_noisy = convmtx2_valid(I_hat, L_alg, K_alg)
    if weight_choice == '1':
        # weight matrix version I
        weight_mtx = 1. / (blk_sum(np.ones(anni_mtx_noisy.shape),
                                   blk_sz0, blk_sz1, num_blk0, num_blk1)
                           )
    elif weight_choice == '2':
        # weight matrix version II
        freq_rescale_weights = np.abs(convmtx2_valid(np.reshape(freq_scaling, b.shape, order='F'), L_alg, K_alg))
        freq_rescale_weights[freq_rescale_weights < 1e-10] += 1e-3  # avoid dividing by 0
        weight_mtx = 1. / (freq_rescale_weights *
                           blk_sum(np.ones(anni_mtx_noisy.shape),
                                   blk_sz0, blk_sz1, num_blk0, num_blk1)
                           )
    else:
        # weight matrix version III (equal weights)
        weight_mtx = np.ones(anni_mtx_noisy.shape)

    # initialise annihilation data matrix
    anni_mtx_denoised = anni_mtx_noisy.copy()
    mats = anni_mtx_denoised.copy()  # auxiliary matrix
    for loop in range(max_iter):
        U_loop, s_loop, Vh_loop = \
            linalg.svd(mats + gamma * (anni_mtx_denoised - mats) +
                       mu * (anni_mtx_noisy - anni_mtx_denoised) * weight_mtx,
                       full_matrices=False, compute_uv=True)
        # thresholding singular values
        s_loop[-1 - num_zero + 1::] = 0.
        # re-synthesize the matrix
        anni_mtx_denoised = np.dot(U_loop, np.dot(np.diag(s_loop), Vh_loop))
        mats = mats - anni_mtx_denoised + \
               blk_avg(2 * anni_mtx_denoised - mats,
                       blk_sz0, blk_sz1, num_blk0, num_blk1)

    anni_mtx_denoised = blk_avg(anni_mtx_denoised, blk_sz0, blk_sz1, num_blk0, num_blk1)

    # the denoised Fourier data
    I_hat_denoised = get_mtx_entries(anni_mtx_denoised, blk_sz0, blk_sz1, num_blk0, num_blk1)

    # build the associated convolution matrix from the denoised data
    anni_mtx_denoised_r = convmtx2_valid(np.real(I_hat_denoised), L, K)
    anni_mtx_denoised_i = convmtx2_valid(np.imag(I_hat_denoised), L, K)
    anni_mtx_denoised_ri_D = np.dot(np.vstack((np.hstack((anni_mtx_denoised_r,
                                                          -anni_mtx_denoised_i)),
                                               np.hstack((anni_mtx_denoised_i,
                                                          anni_mtx_denoised_r))
                                               )),
                                    D)
    # take svd decomposition of the annihilating filter matrix and extract
    # the singular vector that has the smallest singular value
    Vh = linalg.svd(anni_mtx_denoised_ri_D, compute_uv=True)[2]
    c_h_ri = Vh.conj().T[:, -1]
    coef_recon = np.reshape(np.dot(D1_jD2, c_h_ri), (L, K), order='F')
    return coef_recon


def blk_sum(mtx, blk_sz0, blk_sz1, num_blk0, num_blk1):
    """
    average a matrix so that it satisfies the block Toeplitz structure
    :param mtx: the matrix to be averaged
    :param blk_sz0: block size (the vertical dimension)
    :param blk_sz1: block size (the horizontal dimension)
    :param num_blk0: number of blocks (the vertical dimension)
    :param num_blk1: number of blocks (the horizontal dimension)
    :return:
    """
    # initialise the average matrix
    mtx_avg = np.zeros(mtx.shape, dtype=mtx.dtype)
    idx_h, idx_v = np.meshgrid(np.arange(num_blk1), np.arange(num_blk0))
    # parse the blocks
    for count in range(-num_blk0 + 1, num_blk1):
        idx_h_count = np.diag(idx_h, count)
        idx_v_count = np.diag(idx_v, count)
        sum_mtx = np.zeros((blk_sz0, blk_sz1), dtype=mtx.dtype)

        # first average block-wise
        for inner in range(idx_h_count.size):
            idx_h0 = idx_h_count[inner] * blk_sz1
            idx_v0 = idx_v_count[inner] * blk_sz0
            sum_mtx += mtx[idx_v0:idx_v0 + blk_sz0, idx_h0:idx_h0 + blk_sz1]

        # now average the entries in the averaged block matrix
        sum_blk = diag_sum(sum_mtx, blk_sz0, blk_sz1)

        # assign the summed matrix to the output
        for inner in range(idx_h_count.size):
            idx_h0 = idx_h_count[inner] * blk_sz1
            idx_v0 = idx_v_count[inner] * blk_sz0
            mtx_avg[idx_v0:idx_v0 + blk_sz0, idx_h0:idx_h0 + blk_sz1] = sum_blk
    return mtx_avg


def diag_sum(mtx, mtx_sz0, mtx_sz1):
    """
    average a matrix so that it conforms to the Toeplitz structure
    :param mtx: the matrix to be averaged
    :param mtx_sz0: size of the matrix (vertical dimension)
    :param mtx_sz1: size of the matrix (horizontal dimension)
    :return:
    """
    col = np.zeros(mtx_sz0, dtype=mtx.dtype)
    row = np.zeros(mtx_sz1, dtype=mtx.dtype)
    for count in range(0, -mtx_sz0, -1):
        col[-count] = np.sum(np.diag(mtx, count))
    for count in range(mtx_sz1):
        row[count] = np.sum(np.diag(mtx, count))
    return linalg.toeplitz(col, row)


# ============ for Cadzow denoising method ============= #
def cadzow_iter_fri_curve(G, a, K, L, K_cad, L_cad, B_x, B_y, tau_x, tau_y, max_iter=100):
    """
    cadzow iterative denoising method for FRI curves.
    :param G: the linear mapping between the spatial domain samples and the Fourier samples.
              Typically a truncated DFT transformation.
    :param a: the spatial domain samples
    :param K: size of the curve coefficients along x-axis (2K_0 + 1)
    :param L: size of the curve coefficients along y-axis (2L_0 + 1)
    :param K_cad: size of the assumed curve coefficients along x-axis,
                  which is used in the Cadzow iterative denoising. It should be at least K.
    :param L_cad: size of the assumed curve coefficients along y-axis,
                  # which is used in the Cadzow iterative denoising. It should be at least L.
    :param B_x: bandwidth of the lowpass filter along x-axis
    :param B_y: bandwidth of the lowpass filter along y-axis
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :param max_iter: maximum number of iterations
    :return:
    """
    k_limit = np.int(np.floor(B_x * tau_x / 2.))
    l_limit = np.int(np.floor(B_y * tau_y / 2.))
    k_grid, l_grid = np.meshgrid(np.arange(-k_limit, k_limit + 1),
                                 np.arange(-l_limit, l_limit + 1))
    k_grid = np.reshape(k_grid, (-1, 1), order='F')
    l_grid = np.reshape(l_grid, (-1, 1), order='F')

    sz_b0 = 2 * l_limit + 1
    sz_b1 = 2 * k_limit + 1
    b = np.reshape(linalg.lstsq(G, np.reshape(a, (-1, 1), order='F'))[0],
                   (sz_b0, sz_b1), order='F')

    # number of zeros of the annihilating filter matrix in the Cadzow iteration
    num_zero = (K_cad - K + 1) * (L_cad - L + 1)
    blk_sz0 = b.shape[0] - L_cad + 1
    blk_sz1 = L_cad
    num_blk0 = b.shape[1] - K_cad + 1
    num_blk1 = K_cad

    # the scaling factor in the annihilation constraint
    freq_scaling = 2 * np.pi / tau_x * k_grid + 1j * 2 * np.pi / tau_y * l_grid

    # take the Hermtian symmetry into account
    numel_coef_h = np.int(np.floor(K * L / 2))
    D1, D2 = hermitian_expansion(numel_coef_h + 1)

    D = np.vstack((np.hstack((D1, np.zeros((2 * numel_coef_h + 1, numel_coef_h), dtype=float))),
                   np.hstack((np.zeros((2 * numel_coef_h + 1, numel_coef_h + 1), dtype=float), D2))
                   ))

    D1_jD2 = np.hstack((D1, 1j * D2))

    I_hat = b * np.reshape(freq_scaling, b.shape, order='F')
    anni_mtx = convmtx2_valid(I_hat, L_cad, K_cad)

    for loop in range(max_iter):
        U_loop, s_loop, Vh_loop = linalg.svd(anni_mtx, full_matrices=False, compute_uv=True)

        if s_loop[-1 - num_zero] / s_loop[-1 - num_zero + 1] > 1e12:
            break

        # thresholding singular values
        s_loop[-1 - num_zero + 1::] = 0.

        # re-synthesize the matrix
        anni_mtx_threshold = np.dot(U_loop, np.dot(np.diag(s_loop), Vh_loop))
        anni_mtx = blk_avg(anni_mtx_threshold, blk_sz0, blk_sz1, num_blk0, num_blk1)

    # the denoised Fourier data
    I_hat_denoised = get_mtx_entries(anni_mtx, blk_sz0, blk_sz1, num_blk0, num_blk1)
    # build the associated convolution matrix from the denoised data
    anni_mtx_denoised_r = convmtx2_valid(np.real(I_hat_denoised), L, K)
    anni_mtx_denoised_i = convmtx2_valid(np.imag(I_hat_denoised), L, K)
    anni_mtx_denoised_ri_D = np.dot(np.vstack((np.hstack((anni_mtx_denoised_r,
                                                          -anni_mtx_denoised_i)),
                                               np.hstack((anni_mtx_denoised_i,
                                                          anni_mtx_denoised_r))
                                               )),
                                    D)
    # take svd decomposition of the annihilating filter matrix and extract
    # the singular vector that has the smallest singular value
    Vh = linalg.svd(anni_mtx_denoised_ri_D, compute_uv=True)[2]
    c_h_ri = Vh.conj().T[:, -1]
    coef_recon = np.reshape(np.dot(D1_jD2, c_h_ri), (L, K), order='F')
    return coef_recon


def blk_avg(mtx, blk_sz0, blk_sz1, num_blk0, num_blk1):
    """
    average a matrix so that it satisfies the block Toeplitz structure
    :param mtx: the matrix to be averaged
    :param blk_sz0: block size (the vertical dimension)
    :param blk_sz1: block size (the horizontal dimension)
    :param num_blk0: number of blocks (the vertical dimension)
    :param num_blk1: number of blocks (the horizontal dimension)
    :return:
    """
    # initialise the average matrix
    mtx_avg = np.zeros(mtx.shape, dtype=mtx.dtype)
    idx_h, idx_v = np.meshgrid(np.arange(num_blk1), np.arange(num_blk0))
    # parse the blocks
    for count in range(-num_blk0 + 1, num_blk1):
        idx_h_count = np.diag(idx_h, count)
        idx_v_count = np.diag(idx_v, count)
        sum_mtx = np.zeros((blk_sz0, blk_sz1), dtype=mtx.dtype)
        # first average block-wise
        for inner in range(idx_h_count.size):
            idx_h0 = idx_h_count[inner] * blk_sz1
            idx_v0 = idx_v_count[inner] * blk_sz0
            sum_mtx += mtx[idx_v0:idx_v0 + blk_sz0, idx_h0:idx_h0 + blk_sz1]
        avg_blk = sum_mtx / idx_h_count.size
        # now average the entries in the averaged block matrix
        avg_blk = diag_avg(avg_blk, blk_sz0, blk_sz1)
        # assign the averaged matrix to the output
        for inner in range(idx_h_count.size):
            idx_h0 = idx_h_count[inner] * blk_sz1
            idx_v0 = idx_v_count[inner] * blk_sz0
            mtx_avg[idx_v0:idx_v0 + blk_sz0, idx_h0:idx_h0 + blk_sz1] = avg_blk
    return mtx_avg


def diag_avg(mtx, mtx_sz0, mtx_sz1):
    """
    average a matrix so that it conforms to the Toeplitz structure
    :param mtx: the matrix to be averaged
    :param mtx_sz0: size of the matrix (vertical dimension)
    :param mtx_sz1: size of the matrix (horizontal dimension)
    :return:
    """
    col = np.zeros(mtx_sz0, dtype=mtx.dtype)
    row = np.zeros(mtx_sz1, dtype=mtx.dtype)
    for count in range(0, -mtx_sz0, -1):
        col[-count] = np.mean(np.diag(mtx, count))
    for count in range(mtx_sz1):
        row[count] = np.mean(np.diag(mtx, count))
    return linalg.toeplitz(col, row)


def get_mtx_entries(mtx, blk_sz0, blk_sz1, num_blk0, num_blk1):
    """
    get the entries of the block-Toeplitz matrix
    :param mtx: the block-Toeplitz matrix
    :param blk_sz0: block size (the vertical dimension)
    :param blk_sz1: block size (the horizontal dimension)
    :param num_blk0: number of blocks (the vertical dimension)
    :param num_blk1: number of blocks (the horizontal dimension)
    :return:
    """
    data = np.zeros((blk_sz0 + blk_sz1 - 1,
                     num_blk0 + num_blk1 - 1), dtype=mtx.dtype)
    for count in range(num_blk1 - 1, -1, -1):
        mtx_count = mtx[0:blk_sz0, count * blk_sz1:(count + 1) * blk_sz1]
        data[:, -count + num_blk1 - 1] = np.concatenate((mtx_count[0, -1::-1],
                                                         mtx_count[1::, 0]))
    for count in range(1, num_blk0):
        mtx_count = mtx[count * blk_sz0:(count + 1) * blk_sz0, 0:blk_sz1]
        data[:, count + num_blk1 - 1] = np.concatenate((mtx_count[0, -1::-1],
                                                        mtx_count[1::, 0]))
    return data


def lsq_fri_curve(G, a, K, L, B_x, B_y, tau_x, tau_y):
    k_limit = np.int(np.floor(B_x * tau_x / 2.))
    l_limit = np.int(np.floor(B_y * tau_y / 2.))
    k_grid, l_grid = np.meshgrid(np.arange(-k_limit, k_limit + 1),
                                 np.arange(-l_limit, l_limit + 1))
    k_grid = np.reshape(k_grid, (-1, 1), order='F')
    l_grid = np.reshape(l_grid, (-1, 1), order='F')

    sz_b0 = 2 * l_limit + 1
    sz_b1 = 2 * k_limit + 1

    b = np.reshape(linalg.lstsq(G, np.reshape(a, (-1, 1), order='F'))[0],
                   (sz_b0, sz_b1), order='F')

    # the scaling factor in the annihilation constraint
    freq_scaling = 2 * np.pi / tau_x * k_grid + 1j * 2 * np.pi / tau_y * l_grid
    numel_coef_h = np.int(np.floor(K * L / 2))
    D1, D2 = hermitian_expansion(numel_coef_h + 1)

    D = np.vstack((np.hstack((D1, np.zeros((2 * numel_coef_h + 1, numel_coef_h), dtype=float))),
                   np.hstack((np.zeros((2 * numel_coef_h + 1, numel_coef_h + 1), dtype=float), D2))
                   ))
    D1_jD2 = np.hstack((D1, 1j * D2))

    I_hat = b * np.reshape(freq_scaling, b.shape, order='F')

    anni_mtx_r = convmtx2_valid(np.real(I_hat), L, K)
    anni_mtx_i = convmtx2_valid(np.imag(I_hat), L, K)
    anni_mtx_ri_D = np.dot(np.vstack((np.hstack((anni_mtx_r, -anni_mtx_i)),
                                      np.hstack((anni_mtx_i, anni_mtx_r))
                                      )), D)
    Vh = linalg.svd(anni_mtx_ri_D, compute_uv=True)[2]
    c_h_ri = Vh.conj().T[:, -1]
    coef = np.reshape(np.dot(D1_jD2, c_h_ri), (L, K), order='F')

    return coef


# ============= for batch run ============== #
def run_algs(coef, G, samples_noiseless, P, K, L, K_cad, L_cad, B_x, B_y, tau_x, tau_y,
             max_iter_cadzow=1000, max_iter_srla=1000, max_ini=50):
    """
    run both the Cadzow method and the proposed method with the same set of samples
    :param coef: the ground truth curve coefficients (used for comparison)
    :param G: linear mapping between the spatial domain samples and the FRI sequence
    :param samples_noiseless: the noiseless samples
    :param P: the noise level (SNR in dB)
    :param K: the dimension of the curve coeffiicents are L x K
    :param L: the dimension of the curve coeffiicents are L x K
    :param K_cad: size of the assumed curve coefficients along x-axis,
                  which is used in the Cadzow iterative denoising. It should be at least K.
    :param L_cad: size of the assumed curve coefficients along y-axis,
                  # which is used in the Cadzow iterative denoising. It should be at least L.
    :param B_x: bandwidth of the lowpass filter along x-axis
    :param B_y: bandwidth of the lowpass filter along y-axis
    :param tau_x: period along x-axis
    :param tau_y: period along y-axis
    :param max_iter_cadzow: maximum number of iterations for Cadzow's method
    :param max_ini: maximum number of initialisations for the proposed method
    :return:
    """
    sp.random.seed()  # so that each subprocess has a different random states
    samples_size = samples_noiseless.shape
    # check whether we are in the case with real-valued samples or not
    real_valued = np.max(np.abs(np.imag(samples_noiseless))) < 1e-12
    # add Gaussian white noise
    if real_valued:
        noise = np.random.randn(samples_size[0], samples_size[1])
        samples_noiseless = np.real(samples_noiseless)
    else:
        noise = np.random.randn(samples_size[0], samples_size[1]) + \
                1j * np.random.randn(samples_size[0], samples_size[1])
    noise = noise / linalg.norm(noise, 'fro') * \
            linalg.norm(samples_noiseless, 'fro') * 10 ** (-P / 20.)
    samples_noisy = samples_noiseless + noise
    # noise energy, in the noiseless case 1e-10 is considered as 0
    noise_level = np.max([1e-10, linalg.norm(noise, 'fro')])

    # least square minimisation
    coef_recon_lsq = lsq_fri_curve(G, samples_noisy, K, L, B_x, B_y, tau_x, tau_y)
    std_lsq = std_normalised(coef_recon_lsq, coef)[0]
    snr_lsq = snr_normalised(coef_recon_lsq, coef)

    # cadzow iterative denoising
    coef_recon_cadzow = cadzow_iter_fri_curve(G, samples_noisy, K, L, K_cad,
                                              L_cad, B_x, B_y, tau_x, tau_y,
                                              max_iter=max_iter_cadzow)
    std_cadzow = std_normalised(coef_recon_cadzow, coef)[0]
    snr_cadzow = snr_normalised(coef_recon_cadzow, coef)

    # structured low rank approximation (SLRA)
    # weight_choice: '1': the default one based on number of the repetition of entries
    # in the block Toeplitz matrix
    # weight_choice: '2': based on the number of repetition of entries in the block
    # Toeplitz matrix and the frequency re-scaling factor in hat_partial_I
    # weight_choice: '3': equal weights for all entries in the block Toeplitz matrix
    coef_recon_slra1 = slra_fri_curve(G, samples_noisy, K, L, K_cad, L_cad,
                                      B_x, B_y, tau_x, tau_y,
                                      max_iter=max_iter_srla,
                                      weight_choice='1')
    std_slra = std_normalised(coef_recon_slra1, coef)[0]
    snr_slra = snr_normalised(coef_recon_slra1, coef)

    # the proposed approach
    xhat_recon, min_error, coef_recon, ini = \
        recon_fri_curve(G, samples_noisy, K, L,
                        B_x, B_y, tau_x, tau_y, noise_level,
                        max_ini, stop_cri='max_iter')

    std_proposed = std_normalised(coef_recon, coef)[0]
    snr_proposed = snr_normalised(coef_recon, coef)
    return std_lsq, snr_lsq, std_cadzow, snr_cadzow, \
           std_slra, snr_slra, \
           std_proposed, snr_proposed


# ============= for radio astronomy problem =========== #
def dirac_2d_v_and_h(direction, G_row, vec_len_row, num_vec_row,
                     G_col, vec_len_col, num_vec_col,
                     a, K, noise_level, max_ini, stop_cri):
    """
    used to run the reconstructions along horizontal and vertical directions in parallel.
    """
    if direction == 0:  # row reconstruction
        c_recon, min_error, b_recon, ini = \
            recon_2d_dirac_vertical(G_row, vec_len_row, num_vec_row,
                                    a, K, noise_level, max_ini, stop_cri)
    else:  # column reconstruction
        c_recon, min_error, b_recon, ini = \
            recon_2d_dirac_vertical(G_col, vec_len_col, num_vec_col,
                                    a, K, noise_level, max_ini, stop_cri)
    return c_recon, min_error, b_recon, ini


def recon_2d_dirac(FourierData, K, tau_x, tau_y,
                   tau_inter_x, tau_inter_y, omega_ell, M, N,
                   noise_level, max_ini=100, stop_cri='mse',
                   num_rotation=6):
    """
    :param FourierData: Noisy Fourier transforms of the Diracs at some frequencies
    :param K: number of Dirac
    :param taus: [tau_x, tau_y] space support of the Dirac is [-tau_x/2,tau_x/2] x [-tau_y/2,tau_y/2]
    :param omega_ell: frequencies where the FourierData is taken
    :param M_N: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param noise_level: noise level in the given Fourier data
    :param max_ini: maximum number of random initialisation allowed in the IQML algorithm
    :param stop_cri: stopping criteria: 1) 'mse' (default) or 2) 'maxiter'
    :param exhaustive_search: whether or not to use exhaustive search to determine the correct (x,y) combinations
    :return: xk_recon: reconstructed horizontal position of the Dirac
             yk_recon: reconstructed vertical position of the Dirac
             ak_recon: reconstructed amplitudes of the Dirac
    """
    # omega_ell is an L by 2 matrix. The FIRST column corresponds to the HORIZONTAL (x-axis) frequencies where the
    # Fourier measurements are taken; while the SECOND column corresponds to the VERTICAL (y-axis) frequencies.
    omega_ell_x = omega_ell[:, 0]
    omega_ell_y = omega_ell[:, 1]

    # verify input parameters
    # interpolation points cannot be too far apart
    assert tau_inter_x >= tau_x and tau_inter_y >= tau_y
    # M*tau is an odd number
    assert M * tau_inter_x % 2 == 1 and N * tau_inter_y % 2 == 1
    # G is a tall matrix
    assert M * tau_inter_x * N * tau_inter_y <= omega_ell_x.size
    # minimum number of annihilation equations compared with
    # the size of annihilating filter coefficients
    assert (M * tau_inter_x - K) * N * tau_inter_y >= K + 1 and \
           (N * tau_inter_y - K) * M * tau_inter_x >= K + 1

    tau_inter_x = np.float(tau_inter_x)
    tau_inter_y = np.float(tau_inter_y)

    m_limit = np.int(np.floor(M * tau_inter_x / 2.))  # M*tau_inter_x needs to be an ODD number
    n_limit = np.int(np.floor(N * tau_inter_y / 2.))  # N*tau_inter_y needs to be an ODD number

    # build the linear matrix that relates the Fourier transform on a uniform grid to the given Fourier domain
    # measurements 'a' Gln is used when annihilation is applied for each COLUMN!!
    m_len = 2 * m_limit + 1
    n_len = 2 * n_limit + 1

    # initialisation
    min_error_rotate = float('inf')
    xk_opt = np.zeros(K)
    yk_opt = np.zeros(K)
    ak_opt = np.zeros(K)
    angles_seq = np.linspace(0, np.pi, num=num_rotation, endpoint=False)
    for rand_rotate in angles_seq:
        # rotate a random angle
        theta = rand_rotate + np.pi / num_rotation * np.random.rand()
        omega_ell_x_rotated = np.cos(theta) * omega_ell_x - np.sin(theta) * omega_ell_y
        omega_ell_y_rotated = np.sin(theta) * omega_ell_x + np.cos(theta) * omega_ell_y
        Gln = build_G1d(omega_ell_x_rotated, omega_ell_y_rotated, M, N, tau_inter_x, tau_inter_y)
        Glm = build_G1d(omega_ell_y_rotated, omega_ell_x_rotated, N, M, tau_inter_y, tau_inter_x)
        # annihilating filter coefficients when the constraint is imposed for each COLUMN
        if not (stop_cri == 'mse'):
            c_recon_col, min_error_col, b_recon_col, ini_col = \
                recon_2d_dirac_vertical_parallel(Gln, n_len, m_len, FourierData, K, max_ini)
            c_recon_row, min_error_row, b_recon_row, ini_row = \
                recon_2d_dirac_vertical_parallel(Glm, m_len, n_len, FourierData, K, max_ini)
        else:
            # the parallel version II
            partial_dirac_2d = partial(dirac_2d_v_and_h,
                                       G_row=Gln, vec_len_row=n_len, num_vec_row=m_len,
                                       G_col=Glm, vec_len_col=m_len, num_vec_col=n_len,
                                       a=FourierData, K=K, noise_level=noise_level,
                                       max_ini=max_ini, stop_cri=stop_cri)
            anni_res = Parallel(n_jobs=2)(
                delayed(partial_dirac_2d)(ins) for ins in range(2))
            c_recon_col, min_error_col, b_recon_col, ini_col = anni_res[0]
            c_recon_row, min_error_row, b_recon_row, ini_row = anni_res[1]

        vk = np.roots(np.squeeze(c_recon_col))
        vk /= np.abs(vk)
        yk_recon = np.real(1j * tau_inter_y / (2. * np.pi) * np.log(vk))

        uk = np.roots(np.squeeze(c_recon_row))
        uk /= np.abs(uk)
        xk_recon = np.real(1j * tau_inter_x / (2. * np.pi) * np.log(uk))

        xk_recon_grids, yk_recon_grids = np.meshgrid(xk_recon, yk_recon)
        Phi_amp = np.exp(-1j * omega_ell_x_rotated[:, np.newaxis] *
                         np.reshape(xk_recon_grids, (1, -1), order='F') -
                         1j * omega_ell_y_rotated[:, np.newaxis] *
                         np.reshape(yk_recon_grids, (1, -1), order='F'))

        # the amplitudes are all positive
        FourierData_ri = np.concatenate((np.real(FourierData), np.imag(FourierData)))
        ak_recon = sp.optimize.nnls(np.vstack((np.real(Phi_amp), np.imag(Phi_amp))),
                                    FourierData_ri)[0]
        # find the K largest amplitudes among the K^2 possible locations
        ak_sort_idx = (np.argsort(np.abs(ak_recon)))[:-(K + 1):-1]
        # exatract the correct x-coordinates and y-coordinates with the correct associations
        xk_recon = (xk_recon_grids.flatten('F'))[ak_sort_idx]
        yk_recon = (yk_recon_grids.flatten('F'))[ak_sort_idx]

        # rotate back
        xk_rotate_back = np.cos(theta) * xk_recon + np.sin(theta) * yk_recon
        yk_rotate_back = -np.sin(theta) * xk_recon + np.cos(theta) * yk_recon
        xk_recon = xk_rotate_back
        yk_recon = yk_rotate_back

        # use the correctly combined (xk, yk)'s to reconstruct amplitudes
        Phi_sorted = np.exp(-1j * omega_ell_x[:, np.newaxis] *
                            np.reshape(xk_recon, (1, -1), order='F') -
                            1j * omega_ell_y[:, np.newaxis]
                            * np.reshape(yk_recon, (1, -1), order='F'))
        # ak_recon = linalg.lstsq(Phi_sorted, FourierData)[0]
        ak_recon = sp.optimize.nnls(np.vstack((np.real(Phi_sorted), np.imag(Phi_sorted))),
                                    FourierData_ri)[0]
        error_loop = linalg.norm(FourierData - np.dot(Phi_sorted, ak_recon))

        if error_loop < min_error_rotate:
            min_error_rotate = error_loop
            xk_opt = xk_recon
            yk_opt = yk_recon
            ak_opt = ak_recon

    return xk_opt, yk_opt, ak_opt


def recon_2d_dirac_vertical_parallel(Gln, vec_len, num_vec, a, K, max_ini=100):
    """
    The PARALLEL version of recon_2d_dirac_vertical
    Gln is a matrix that groups Gln horizontally for different m's
    vec_len:  length of each vector that should satisfy the annihilation constraint
    num_vec:  total number of such vectors that should be annihilated by the same filter,
              whose coefficients are given by 'c'
    a:        given Fourier domain measurements taken at L different frequencies
    K:        number of Dirac
    noise_level: allowed noise level
    max_ini:  maximum number of random initialisation
    """
    beta = linalg.lstsq(Gln, a)[0]  # the least square solution
    beta_reshape = np.reshape(beta, (vec_len, num_vec), order='F')
    Tbeta = np.zeros((num_vec * (vec_len - K), K + 1), dtype=beta.dtype)
    for loop in range(num_vec):
        Tbeta[loop * (vec_len - K): (loop + 1) * (vec_len - K), :] = \
            linalg.toeplitz(beta_reshape[K:, loop], beta_reshape[K::-1, loop])
    col_anni_res = Parallel(n_jobs=cpu_count() - 1)(
        delayed(coef_ini_loop_parallel)(Tbeta, Gln, a, vec_len, num_vec, K) for ini in range(max_ini))
    min_error_all = [col_extract[-1] for col_extract in col_anni_res]

    min_idx = np.array(min_error_all).argmin()
    c_opt_col = col_anni_res[min_idx][0]
    b_opt_col = col_anni_res[min_idx][1]
    min_error = min_error_all[min_idx]
    ini = max_ini - 1  # indexing from 0
    return c_opt_col, min_error, b_opt_col, ini


def recon_2d_dirac_vertical(Gln, vec_len, num_vec, a, K, noise_level, max_ini=100, stop_cri='mse'):
    """
    Gln is a matrix that groups Gln horizontally for different m's
    vec_len:  length of each vector that should satisfy the annihilation constraint
    num_vec:  total number of such vectors that should be annihilated by the same filter,
              whose coefficients are given by 'c'
    a:        given Fourier domain measurements taken at L different frequencies
    K:        number of Dirac
    noise_level: allowed noise level
    max_ini:  maximum number of random initialisation
    """
    compute_mse = (stop_cri == 'mse')
    nm_len = vec_len * num_vec
    m_times_n_K = num_vec * (vec_len - K)
    GtG = np.dot(Gln.conj().T, Gln)
    Gt_a = np.dot(Gln.conj().T, a)

    max_iter = 50
    min_error = float('inf')
    # beta = linalg.solve(GtG, Gt_a)
    beta = linalg.lstsq(Gln, a)[0]  # the least square solution incase G is rank deficient
    beta_reshape = np.reshape(beta, (vec_len, num_vec), order='F')
    Tbeta = np.zeros((m_times_n_K, K + 1), dtype=beta.dtype)
    rep_mtx = np.eye(num_vec)
    for loop in range(num_vec):
        Tbeta[loop * (vec_len - K): (loop + 1) * (vec_len - K), :] = \
            linalg.toeplitz(beta_reshape[K::, loop], beta_reshape[K::-1, loop])

    rhs = np.concatenate((np.zeros(K + 1 + m_times_n_K + nm_len, dtype=float),
                          np.array(1, dtype=float)[np.newaxis]))
    rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
                                   np.zeros((m_times_n_K, 1), dtype=Gt_a.dtype))))

    for ini in range(max_ini):
        c = np.random.randn(K + 1) + 1j * np.random.randn(K + 1)
        c0 = c[:, np.newaxis].copy()
        error_seq = np.zeros(max_iter)
        R_loop = np.kron(rep_mtx,
                         linalg.toeplitz(np.concatenate((np.array([c[-1]]),
                                                         np.zeros(vec_len - K - 1, dtype=complex))),
                                         np.concatenate((c[::-1],
                                                         np.zeros(vec_len - K - 1, dtype=complex)))
                                         )
                         )
        # first row in Mtx_loop
        Mtx_loop_first_row = np.hstack((np.zeros((K + 1, K + 1)), Tbeta.conj().T, np.zeros((K + 1, nm_len)), c0))
        # last row in Mtx_loop
        Mtx_loop_last_row = np.hstack((c0.conj().T, np.zeros((1, m_times_n_K + nm_len + 1))))

        for inner in range(max_iter):
            Mtx_loop = np.vstack((Mtx_loop_first_row,
                                  np.hstack((Tbeta, np.zeros((m_times_n_K, m_times_n_K)), -R_loop,
                                             np.zeros((m_times_n_K, 1)))),
                                  np.hstack((np.zeros((nm_len, K + 1)), -R_loop.conj().T, GtG, np.zeros((nm_len, 1)))),
                                  Mtx_loop_last_row
                                  ))
            # matrix should be Hermitian symmetric
            Mtx_loop += Mtx_loop.conj().T
            Mtx_loop *= 0.5
            c = linalg.solve(Mtx_loop, rhs, check_finite=False)[0:K + 1]

            R_loop = np.kron(rep_mtx,
                             linalg.toeplitz(np.concatenate((np.array([c[-1]]),
                                                             np.zeros(vec_len - K - 1, dtype=complex))),
                                             np.concatenate((c[::-1],
                                                             np.zeros(vec_len - K - 1, dtype=complex)))
                                             )
                             )
            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.conj().T)),
                                    np.hstack((R_loop, np.zeros((m_times_n_K, m_times_n_K))))
                                    ))
            # matrix should be Hermitian symmetric
            Mtx_brecon += Mtx_brecon.conj().T
            Mtx_brecon *= 0.5
            b_recon = linalg.solve(Mtx_brecon, rhs_bl, check_finite=False)[0:nm_len]

            error_seq[inner] = linalg.norm(a - np.dot(Gln, b_recon))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt_col = b_recon
                c_opt_col = c
            if min_error < noise_level and compute_mse:
                break
        if min_error < noise_level and compute_mse:
            break
    return c_opt_col, min_error, b_opt_col, ini


def coef_ini_loop_parallel(Tbeta, Gln, a, vec_len, num_vec, K):
    sp.random.seed()  # so that each parallel process has a different random state
    max_iter = 50
    nm_len = vec_len * num_vec
    m_times_n_K = num_vec * (vec_len - K)
    min_error = float('inf')

    Gt_a = np.dot(Gln.conj().T, a)
    GtG = np.dot(Gln.conj().T, Gln)
    rhs = np.append(np.zeros(K + 1 + m_times_n_K + nm_len, dtype=float), 1)
    rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
                                   np.zeros((m_times_n_K, 1), dtype=Gt_a.dtype))))

    c = np.random.randn(K + 1) + 1j * np.random.randn(K + 1)
    c0 = c[:, np.newaxis].copy()

    R_loop = linalg.block_diag(*([linalg.toeplitz(np.concatenate((np.array([c[-1]]),
                                                                  np.zeros(vec_len - K - 1, dtype=complex))),
                                                  np.concatenate((c[::-1],
                                                                  np.zeros(vec_len - K - 1, dtype=complex)))
                                                  )] * num_vec))
    # first row in Mtx_loop
    Mtx_loop_first_row = np.hstack((np.zeros((K + 1, K + 1)), Tbeta.conj().T, np.zeros((K + 1, nm_len)), c0))

    # last row in Mtx_loop
    Mtx_loop_last_row = np.hstack((c0.conj().T, np.zeros((1, m_times_n_K + nm_len + 1))))

    idx_row_s = K + 1
    idx_row_e = idx_row_s + m_times_n_K
    idx_col_s = K + 1 + m_times_n_K
    idx_col_e = idx_col_s + nm_len
    for inner in range(max_iter):
        if inner == 0:
            Mtx_loop = np.vstack((Mtx_loop_first_row,
                                  np.hstack((Tbeta, np.zeros((m_times_n_K, m_times_n_K)), -R_loop,
                                             np.zeros((m_times_n_K, 1)))),
                                  np.hstack((np.zeros((nm_len, K + 1)), -R_loop.conj().T, GtG, np.zeros((nm_len, 1)))),
                                  Mtx_loop_last_row
                                  ))
            Mtx_loop += Mtx_loop.conj().T
            Mtx_loop *= 0.5
        else:
            Mtx_loop[idx_row_s:idx_row_e, idx_col_s:idx_col_e] = -R_loop
            Mtx_loop[idx_col_s:idx_col_e, idx_row_s:idx_row_e] = -R_loop.conj().T

        c = linalg.solve(Mtx_loop, rhs)[:K + 1]

        R_loop = linalg.block_diag(*([linalg.toeplitz(np.concatenate((np.array([c[-1]]),
                                                                      np.zeros(vec_len - K - 1, dtype=complex))),
                                                      np.concatenate((c[::-1],
                                                                      np.zeros(vec_len - K - 1, dtype=complex)))
                                                      )] * num_vec))

        if inner == 0:
            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.conj().T)),
                                    np.hstack((R_loop, np.zeros((m_times_n_K, m_times_n_K))))
                                    ))
            Mtx_brecon += Mtx_brecon.conj().T
            Mtx_brecon *= 0.5
        else:
            Mtx_brecon[:nm_len, nm_len:] = R_loop.conj().T
            Mtx_brecon[nm_len:, :nm_len] = R_loop

        b_recon = linalg.solve(Mtx_brecon, rhs_bl)[:nm_len]
        error_seq_loop = linalg.norm(a - np.dot(Gln, b_recon))
        if error_seq_loop < min_error:
            min_error = error_seq_loop
            b_opt = b_recon
            c_opt = c
    return c_opt, b_opt, min_error

# def coef_ini_loop_parallel(Tbeta, Gln, a, vec_len, num_vec, K):
#     sp.random.seed()  # so that each parallel process has a different random state
#     max_iter = 50
#     nm_len = vec_len * num_vec
#     m_times_n_K = num_vec * (vec_len - K)
#     min_error = float('inf')
#
#     rep_mtx = np.eye(num_vec)
#     GtG = np.dot(Gln.conj().T, Gln)
#     Gt_a = np.dot(Gln.conj().T, a)
#     rhs = np.concatenate((np.zeros(K + 1 + m_times_n_K + nm_len, dtype=float),
#                           np.array(1, dtype=float)[np.newaxis]))
#     rhs_bl = np.squeeze(np.vstack((Gt_a[:, np.newaxis],
#                                    np.zeros((m_times_n_K, 1), dtype=Gt_a.dtype))))
#
#     c = np.random.randn(K + 1) + 1j * np.random.randn(K + 1)
#     c0 = c[:, np.newaxis].copy()
#     error_seq = np.zeros(max_iter)
#     R_loop = np.kron(rep_mtx,
#                      linalg.toeplitz(np.concatenate((np.array([c[-1]]),
#                                                      np.zeros(vec_len - K - 1, dtype=complex))),
#                                      np.concatenate((c[::-1],
#                                                      np.zeros(vec_len - K - 1, dtype=complex)))
#                                      )
#                      )
#     # first row in Mtx_loop
#     Mtx_loop_first_row = np.hstack((np.zeros((K + 1, K + 1)), Tbeta.conj().T, np.zeros((K + 1, nm_len)), c0))
#     # last row in Mtx_loop
#     Mtx_loop_last_row = np.hstack((c0.conj().T, np.zeros((1, m_times_n_K + nm_len + 1))))
#
#     for inner in range(max_iter):
#         Mtx_loop = np.vstack((Mtx_loop_first_row,
#                               np.hstack((Tbeta, np.zeros((m_times_n_K, m_times_n_K)), -R_loop,
#                                          np.zeros((m_times_n_K, 1)))),
#                               np.hstack((np.zeros((nm_len, K + 1)), -R_loop.conj().T, GtG, np.zeros((nm_len, 1)))),
#                               Mtx_loop_last_row
#                               ))
#         Mtx_loop += Mtx_loop.conj().T
#         Mtx_loop *= 0.5
#
#         c = linalg.solve(Mtx_loop, rhs)[0:K + 1]
#
#         R_loop = np.kron(rep_mtx,
#                          linalg.toeplitz(np.concatenate((np.array([c[-1]]),
#                                                          np.zeros(vec_len - K - 1, dtype=complex))),
#                                          np.concatenate((c[::-1],
#                                                          np.zeros(vec_len - K - 1, dtype=complex)))
#                                          )
#                          )
#         Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.conj().T)),
#                                 np.hstack((R_loop, np.zeros((m_times_n_K, m_times_n_K))))
#                                 ))
#         Mtx_brecon += Mtx_brecon.conj().T
#         Mtx_brecon *= 0.5
#
#         b_recon = linalg.solve(Mtx_brecon, rhs_bl)[0:nm_len]
#         error_seq[inner] = linalg.norm(a - np.dot(Gln, b_recon))
#         if error_seq[inner] < min_error:
#             min_error = error_seq[inner]
#             b_opt = b_recon
#             c_opt = c
#     return c_opt, b_opt, min_error


def build_G1d(omega_ell_x, omega_ell_y, M, N, tau_x, tau_y):
    m_limit = np.int(np.floor(M * tau_x / 2.))  # M*tau1 needs to be an ODD number
    n_limit = np.int(np.floor(N * tau_y / 2.))  # N*tau2 needs to be an ODD number
    m_len = 2 * m_limit + 1
    n_len = 2 * n_limit + 1
    L = omega_ell_x.size
    G = np.zeros((L, n_len * m_len), dtype=float)
    n_seq = np.reshape(np.arange(-n_limit, n_limit + 1), (1, -1), order='F')
    omega_ell_x = np.reshape(omega_ell_x, (-1, 1), order='F')
    omega_ell_y = np.reshape(omega_ell_y, (-1, 1), order='F')
    PhiBase_ln = periodicSinc((tau_y * omega_ell_y - 2. * np.pi * n_seq) / 2., N * tau_y)

    for m_loop in range(-m_limit, m_limit + 1):
        Phi_m = periodicSinc((tau_x * omega_ell_x - 2. * np.pi * m_loop) / 2., M * tau_x)
        G[:, m_len * (m_loop + m_limit): m_len * (m_loop + m_limit + 1)] = PhiBase_ln * Phi_m
    return G


def periodicSinc(t, M):
    numerator = np.sin(t)
    denominator = M * np.sin(t / M)
    if t.size == 1:
        if np.abs(denominator) < 1e-12:
            numerator = np.cos(t)
            denominator = np.cos(t / M)
    else:
        idx = np.abs(denominator) < 1e-12
        numerator[idx] = np.cos(t[idx])
        denominator[idx] = np.cos(t[idx] / M)
    return numerator / denominator


def plot_2d_dirac_loc(xk_recon, yk_recon, alpha_k_recon,
                      xk_ref, yk_ref, alpha_k_ref,
                      K, L, P, tau_x, tau_y,
                      save_figure=False, fig_format='pdf',
                      file_name='recon_2d_dirac_loc.pdf'):
    """
    Plot the reconstructed 2D Diracs along with the original ones.
    :param xk_recon: reconstructed Diracs' horizontal location
    :param yk_recon: reconstructed Diracs' vertical location
    :param alpha_k_recon: reconstructed Diracs' ampltudes
    :param xk_ref: ground truth Diracs' horizontal location
    :param yk_ref: ground truth Diracs' vertical location
    :param alpha_k_ref: ground truth Diracs' amplitudes
    :param K: number of Diracs
    :param L: number of samples
    :param P: noise level in the samples
    :param tau_x: the spatial support of the Diracs along
                  the horizontal direction is [-tau_x/2, tau_x/2]
    :param tau_y: the spatial support of the Diracs along
                  the vertical direction is [-tau_y/2, tau_y/2]
    :param save_figure: whether to save the figure as a file or not
    :param fig_format: figure format
    :param file_name: file nae used to save the figure.
    :return: null
    """
    r_est_error = distance(xk_ref + 1j * yk_ref, xk_recon + 1j * yk_recon)[0]

    plt.figure(figsize=(6, 2.5), dpi=90)
    # subplot 1
    ax1 = plt.axes([0.09, 0.167, 0.308, 0.745])
    subplt22_13_1 = ax1.plot(xk_ref, yk_ref, label='Original Diracs')
    plt.setp(subplt22_13_1, linewidth=1.5,
             color=[0, 0.447, 0.741], mec=[0, 0.447, 0.741], linestyle='None',
             marker='^', markersize=8, markerfacecolor=[0, 0.447, 0.741])
    plt.axis('scaled')
    plt.xlim([-tau_x / 2., tau_x / 2.])
    plt.ylim([-tau_y / 2., tau_y / 2.])

    subplt22_13_2 = plt.plot(xk_recon, yk_recon, hold=True,
                             label='Estimated Diracs')
    plt.setp(subplt22_13_2, linewidth=1.5,
             color=[0.850, 0.325, 0.098], linestyle='None',
             marker='*', markersize=10, markerfacecolor=[0.850, 0.325, 0.098],
             mec=[0.850, 0.325, 0.098])
    plt.xlabel(r'horizontal position $x$', fontsize=12)
    plt.ylabel(r'vertical position $y$', fontsize=12)
    ax1.xaxis.set_label_coords(0.5, -0.11)
    ax1.yaxis.set_label_coords(-0.18, 0.5)
    plt.legend(numpoints=1, loc=0, fontsize=9, framealpha=0.3,
               handletextpad=.2, columnspacing=1.7, labelspacing=0.1)
    plt.title(r'$K={0}$, $L={1}$'.format(repr(K), repr(L)), fontsize=11)

    # subplot 2
    ax2 = plt.axes([0.49, 0.62, 0.505, 0.285])
    subplt22_2_1 = ax2.stem(xk_ref, alpha_k_ref, label='Original Diracs')
    plt.setp(subplt22_2_1[0], marker='^', linewidth=1.5,
             markersize=8, markerfacecolor=[0, 0.447, 0.741],
             mec=[0, 0.447, 0.741])
    plt.setp(subplt22_2_1[1], linewidth=1.5, color=[0, 0.447, 0.741])
    plt.setp(subplt22_2_1[2], linewidth=0)
    plt.xlim([-tau_x / 2., tau_x / 2.])

    subplt22_2_2 = plt.stem(xk_recon, np.real(alpha_k_recon), label='Estimated Diracs', hold=True)
    plt.setp(subplt22_2_2[0], marker='*', linewidth=1.5,
             markersize=10, markerfacecolor=[0.850, 0.325, 0.098],
             mec=[0.850, 0.325, 0.098])
    plt.setp(subplt22_2_2[1], linewidth=1.5, color=[0.850, 0.325, 0.098])
    plt.setp(subplt22_2_2[2], linewidth=0)
    plt.xlim([-tau_x / 2., tau_x / 2.])
    y_min, y_max = plt.ylim()
    if y_min < 0 and y_max > 0:
        plt.ylim([1.1 * y_min, 1.1 * y_max])
    elif y_min >= 0:
        plt.ylim([y_min, 1.1 * y_max])
    elif y_max <= 0:
        plt.ylim([1.1 * y_min, y_max])

    plt.axhline(0, color='k')
    ax2.yaxis.major.locator.set_params(nbins=7)
    plt.xlabel('horizontal position $x$', fontsize=12)
    plt.ylabel('amplitudes', fontsize=12)
    ax2.xaxis.set_label_coords(0.5, -0.28)
    ax2.yaxis.set_label_coords(-0.09, 0.5)
    plt.legend(numpoints=1, loc=1, fontsize=9, framealpha=0.3,
               handletextpad=.2, columnspacing=0.6, labelspacing=0.05, ncol=2)
    r_est_error_pow = np.int(np.floor(np.log10(r_est_error)))
    plt.title(r'$\mbox{{SNR}}={0}$dB, '
              r'$\mathbf{{r}}_{{\mbox{{\footnotesize error}}}}={1:.2f}\times10^{other}$'.format(repr(P),
                                                                                                r_est_error / 10 ** r_est_error_pow,
                                                                                                other='{' + str(
                                                                                                    r_est_error_pow) + '}'),
              fontsize=11)

    # subplot 3
    ax3 = plt.axes([0.49, 0.167, 0.505, 0.285])
    subplt22_3_1 = ax3.stem(yk_ref, alpha_k_ref, label='Original Diracs')
    plt.setp(subplt22_3_1[0], marker='^', linewidth=1.5, markersize=8,
             markerfacecolor=[0, 0.447, 0.741], mec=[0, 0.447, 0.741])
    plt.setp(subplt22_3_1[1], linewidth=1.5, color=[0, 0.447, 0.741])
    plt.setp(subplt22_3_1[2], linewidth=0)
    plt.xlim([-tau_y / 2., tau_y / 2.])

    subplt22_3_2 = plt.stem(yk_recon, np.real(alpha_k_recon),
                            label='Estimated Diracs', hold=True)
    plt.setp(subplt22_3_2[0], marker='*', linewidth=1.5, markersize=10,
             markerfacecolor=[0.850, 0.325, 0.098], mec=[0.850, 0.325, 0.098])
    plt.setp(subplt22_3_2[1], linewidth=1.5, color=[0.850, 0.325, 0.098])
    plt.setp(subplt22_3_2[2], linewidth=0)
    plt.xlim([-tau_y / 2., tau_y / 2.])
    if y_min < 0 and y_max > 0:
        plt.ylim([1.1 * y_min, 1.1 * y_max])
    elif y_min >= 0:
        plt.ylim([y_min, 1.1 * y_max])
    elif y_max <= 0:
        plt.ylim([1.1 * y_min, y_max])

    plt.axhline(0, color='k')
    ax3.yaxis.major.locator.set_params(nbins=7)
    plt.xlabel(r'vertical position $y$', fontsize=12)
    plt.ylabel('amplitudes', fontsize=12)
    ax3.xaxis.set_label_coords(0.5, -0.28)
    ax3.yaxis.set_label_coords(-0.09, 0.5)
    if save_figure:
        plt.savefig(file_name, format=fig_format, dpi=300, transparent=True)
    # plt.show()


def plot_2d_dirac_spec(xk_recon, yk_recon, alpha_k_recon,
                       Ihat_noisy, Ihat_noiseless, omega_ell_x, omega_ell_y,
                       M, N, P, L, show_phase=False, show_dirty_img=False,
                       save_figure=False, fig_format='pdf',
                       file_name='recon_2d_dirac_spec.pdf',
                       **kwargs):
    # resynthesis Fourier transforms at the measured frequencies
    xk_recon_grid, omega_recon_grid_x = np.meshgrid(xk_recon, omega_ell_x)
    yk_recon_grid, omega_recon_grid_y = np.meshgrid(yk_recon, omega_ell_y)
    # Fourier measurements at frequencies omega_ell
    Ihat_recon = np.dot(np.exp(- 1j * omega_recon_grid_x * xk_recon_grid
                               - 1j * omega_recon_grid_y * yk_recon_grid),
                        alpha_k_recon)
    if show_phase:
        plt.figure(figsize=(13, 7), dpi=90)
        # subplot 1
        ax4 = plt.axes([0.06, 0.56, 0.28, 0.4])
        subplt23_1 = ax4.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 c=np.abs(Ihat_noisy), cmap='rainbow')
        plt.colorbar(subplt23_1)
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_x$', fontsize=12)
        plt.ylabel(r'$\omega_y$', fontsize=12)
        plt.title(r'\textit{{Amplitudes}} of noisy samples ($\mbox{{SNR}}={0:.2f}$dB)'.format(P),
                  fontsize=12)

        # subplot 2
        ax5 = plt.axes([0.38, 0.56, 0.28, 0.4])
        subplt23_2 = ax5.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 c=np.abs(Ihat_noiseless), cmap='rainbow')
        plt.colorbar(subplt23_2)
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_x$', fontsize=12)
        plt.ylabel(r'$\omega_y$', fontsize=12)
        plt.title(r'\textit{{Amplitudes}} of noiseless samples ($L={0}$)'.format(repr(L)),
                  fontsize=12)

        # subplot 3
        ax6 = plt.axes([0.70, 0.56, 0.28, 0.4])
        subplt23_3 = ax6.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 vmin=np.abs(Ihat_noiseless).min(), vmax=np.abs(Ihat_noiseless).max(),
                                 c=np.abs(Ihat_recon), cmap='rainbow')
        plt.colorbar(subplt23_3)
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_x$', fontsize=12)
        plt.ylabel(r'$\omega_y$', fontsize=12)
        plt.title(r'\textit{{Amplitudes}} of \textbf{{reconstructed}} samples',
                  fontsize=12)

        # subplot 4
        ax7 = plt.axes([0.06, 0.07, 0.28, 0.4])
        subplt23_4 = ax7.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 vmin=-np.pi, vmax=np.pi,
                                 c=np.angle(Ihat_noisy), cmap='rainbow')
        colorbar23_4 = plt.colorbar(subplt23_4)
        colorbar23_4.set_ticks(np.linspace(-np.pi, np.pi, 9), update_ticks=True)
        colorbar23_4.set_ticklabels([r'$-\pi$', r'$-3\pi/4$',
                                     r'$-\pi/2$', r'$-\pi/4$',
                                     r'$0$', r'$\pi/4$', r'$\pi/2$',
                                     r'$3\pi/4$', r'$\pi$'], update_ticks=True)
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_x$', fontsize=12)
        plt.ylabel(r'$\omega_y$', fontsize=12)
        plt.title(r'\textit{{Phase}} of noisy samples ($\mbox{{SNR}}={0:.2f}$dB)'.format(P),
                  fontsize=12)

        # subplot 5
        ax8 = plt.axes([0.38, 0.07, 0.28, 0.4])
        subplt23_5 = ax8.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 vmin=-np.pi, vmax=np.pi, c=np.angle(Ihat_noiseless),
                                 cmap='rainbow')
        colorbar23_5 = plt.colorbar(subplt23_5)
        colorbar23_5.set_ticks(np.linspace(-np.pi, np.pi, 9), update_ticks=True)
        colorbar23_5.set_ticklabels([r'$-\pi$', r'$-3\pi/4$',
                                     r'$-\pi/2$', r'$-\pi/4$',
                                     r'$0$', r'$\pi/4$', r'$\pi/2$',
                                     r'$3\pi/4$', r'$\pi$'], update_ticks=True)
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_x$', fontsize=12)
        plt.ylabel(r'$\omega_y$', fontsize=12)
        plt.title(r'\textit{{Phase}} of noiseless samples ($L={0}$)'.format(repr(L)),
                  fontsize=12)

        # subplot 6
        ax9 = plt.axes([0.70, 0.07, 0.28, 0.4])
        subplt23_6 = ax9.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 vmin=-np.pi, vmax=np.pi, c=np.angle(Ihat_recon),
                                 cmap='rainbow')
        colorbar23_6 = plt.colorbar(subplt23_6)
        colorbar23_6.set_ticks(np.linspace(-np.pi, np.pi, 9), update_ticks=True)
        colorbar23_6.set_ticklabels([r'$-\pi$', r'$-3\pi/4$',
                                     r'$-\pi/2$', r'$-\pi/4$',
                                     r'$0$', r'$\pi/4$', r'$\pi/2$',
                                     r'$3\pi/4$', r'$\pi$'], update_ticks=True)
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_1$', fontsize=12)
        plt.ylabel(r'$\omega_2$', fontsize=12)
        plt.title(r'\textit{{Phase}} of \textbf{{reconstructed}} samples',
                  fontsize=12)
        if save_figure:
            plt.savefig(file_name, format=fig_format, dpi=300, transparent=True)
        # plt.show()
    elif not save_figure:
        plt.figure(figsize=(8, 2.27), dpi=90)
        if show_dirty_img:
            dirty_img = kwargs['dirty_img']
            num_pixel_y, num_pixel_x = dirty_img.shape
            # subplot 1
            ax4 = plt.axes([0.06, 0.15, 0.285, 0.746])
            plt_dirtyimg = ax4.imshow(np.real(dirty_img), origin='lower', cmap='Spectral_r')
            plt.xticks(np.linspace(num_pixel_x / 12., num_pixel_x - num_pixel_x / 12., num=5),
                       (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
            plt.yticks(np.linspace(num_pixel_y / 12., num_pixel_y - num_pixel_y / 12., num=5),
                       (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
            ax4c = plt.colorbar(plt_dirtyimg, use_gridspec=False,
                                anchor=(-0.15, 0.5), spacing='proportional')
            ax4c.ax.tick_params(labelsize=8.5)
            plt.xlabel(r'horizontal position $x$', fontsize=12)
            plt.ylabel(r'vertical position $y$', fontsize=12)
            ax4.xaxis.set_label_coords(0.5, -0.11)
            ax4.yaxis.set_label_coords(-0.19, 0.5)
            plt.title(r'Dirty Image ' +
                      r'(size: ${0}\times{1}$)'.format(repr(num_pixel_y),
                                                       repr(num_pixel_y)),
                      fontsize=11)

            # subplot 2
            ax5 = plt.axes([0.39, 0.15, 0.285, 0.746])
            subplt23_2 = ax5.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                     c=np.abs(Ihat_noisy), cmap='rainbow')
            plt.colorbar(subplt23_2)
            plt.axis('scaled')
            plt.xlim([-np.pi * M, np.pi * M])
            plt.ylim([-np.pi * N, np.pi * N])
            plt.xlabel(r'$\omega_x$', fontsize=12)
            plt.ylabel(r'$\omega_y$', fontsize=12)
            plt.title(r'Noisy samples ($\mbox{{SNR}}={0:.2f}$dB)'.format(P),
                      fontsize=11)
        else:
            # subplot 1
            ax4 = plt.axes([0.06, 0.15, 0.285, 0.746])
            subplt23_1 = ax4.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                     c=np.abs(Ihat_noisy), cmap='rainbow')
            ax4c = plt.colorbar(subplt23_1, anchor=(-0.3, 0.5))
            plt.axis('scaled')
            ax4.xaxis.set_tick_params(labelsize=8.5)
            ax4.yaxis.set_tick_params(labelsize=8.5)
            ax4c.ax.tick_params(labelsize=8.5)
            plt.xlim([-np.pi * M, np.pi * M])
            plt.ylim([-np.pi * N, np.pi * N])
            plt.xlabel(r'$\omega_1$', fontsize=12)
            plt.ylabel(r'$\omega_2$', fontsize=12)
            ax4.xaxis.set_label_coords(0.5, -0.11)
            ax4.yaxis.set_label_coords(-0.15, 0.5)
            plt.title(r'Noisy Samples ($\mbox{{SNR}}={0}$dB)'.format(repr(P)),
                      fontsize=11)

            # subplot 2
            ax5 = plt.axes([0.39, 0.15, 0.285, 0.746])
            subplt23_2 = ax5.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                     c=np.abs(Ihat_noiseless), cmap='rainbow')
            ax5c = plt.colorbar(subplt23_2, anchor=(-0.3, 0.5))
            plt.axis('scaled')
            ax5.xaxis.set_tick_params(labelsize=8.5)
            ax5.yaxis.set_tick_params(labelsize=8.5)
            ax5c.ax.tick_params(labelsize=8.5)
            plt.xlim([-np.pi * M, np.pi * M])
            plt.ylim([-np.pi * N, np.pi * N])
            plt.xlabel(r'$\omega_1$', fontsize=12)
            plt.ylabel(r'$\omega_2$', fontsize=12)
            ax5.xaxis.set_label_coords(0.5, -0.11)
            ax5.yaxis.set_label_coords(-0.15, 0.5)
            plt.title(r'Noiseless Samples ($L={0}$)'.format(repr(L)), fontsize=11)

        # subplot 3
        ax6 = plt.axes([0.712, 0.15, 0.285, 0.746])
        subplt23_3 = ax6.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 vmin=np.abs(Ihat_noiseless).min(),
                                 vmax=np.abs(Ihat_noiseless).max(),
                                 c=np.abs(Ihat_recon), cmap='rainbow')
        ax6c = plt.colorbar(subplt23_3, anchor=(-0.3, 0.5))
        plt.axis('scaled')
        ax6.xaxis.set_tick_params(labelsize=8.5)
        ax6.yaxis.set_tick_params(labelsize=8.5)
        ax6c.ax.tick_params(labelsize=8.5)
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_1$', fontsize=12)
        plt.ylabel(r'$\omega_2$', fontsize=12)
        ax6.xaxis.set_label_coords(0.5, -0.11)
        ax6.yaxis.set_label_coords(-0.15, 0.5)
        plt.title(r'Reconstructed Samples', fontsize=11)
    else:
        plt.figure(figsize=(3, 3), dpi=90)
        ax4 = plt.axes([0.2, 0.067, 0.75, 0.75])
        subplt23_1 = ax4.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 c=np.abs(Ihat_noisy), cmap='rainbow')
        plt.colorbar(subplt23_1, use_gridspec=False,
                     anchor=(-0.15, 0.5), shrink=0.8, spacing='proportional')
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_x$', fontsize=12)
        plt.ylabel(r'$\omega_y$', fontsize=12)
        plt.title(r'Noisy samples ($\mbox{{SNR}}={0}$dB)'.format(repr(P)),
                  fontsize=11)
        plt.savefig(file_name + r'_noisy.' + fig_format, format=fig_format, dpi=300, transparent=True)
        # plt.show()

        # subplot 2
        plt.figure(figsize=(3, 3), dpi=90)
        ax5 = plt.axes([0.2, 0.067, 0.75, 0.75])
        subplt23_2 = ax5.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 c=np.abs(Ihat_noiseless), cmap='rainbow')
        plt.colorbar(subplt23_2, use_gridspec=False,
                     anchor=(-0.15, 0.5), shrink=0.8, spacing='proportional')
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_x$', fontsize=12)
        plt.ylabel(r'$\omega_y$', fontsize=12)
        plt.title(r'Noiseless samples ($L={0}$)'.format(repr(L)),
                  fontsize=11)
        plt.savefig(file_name + r'_noiseless.' + fig_format, format=fig_format, dpi=300, transparent=True)
        # plt.show()

        # subplot 3
        plt.figure(figsize=(3, 3), dpi=90)
        ax6 = plt.axes([0.2, 0.067, 0.75, 0.75])
        subplt23_3 = ax6.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                                 vmin=np.abs(Ihat_noiseless).min(), vmax=np.abs(Ihat_noiseless).max(),
                                 c=np.abs(Ihat_recon), cmap='rainbow')
        plt.colorbar(subplt23_3, use_gridspec=False,
                     anchor=(-0.15, 0.5), shrink=0.8, spacing='proportional')
        plt.axis('scaled')
        plt.xlim([-np.pi * M, np.pi * M])
        plt.ylim([-np.pi * N, np.pi * N])
        plt.xlabel(r'$\omega_x$', fontsize=12)
        plt.ylabel(r'$\omega_y$', fontsize=12)
        plt.title(r'\phantom{Reconstructed samples}',
                  fontsize=11)
        plt.savefig(file_name + r'_recon.' + fig_format, format=fig_format, dpi=300, transparent=True)
        # plt.show()


def generate_dirty_img(FourierData, omega_ell_x, omega_ell_y,
                       num_pixel_x, num_pixel_y, tau_x, tau_y):
    """
    Generate dirty image from the given Fourier measurements, which are taken at
    irregular frequencies. The image is given by the inverse DFT.
    :param FourierData: the given Fourier transform at certain frequencies
    :param omega_ell_x: frequencies (along x-aixs) where the FourierData is taken
    :param omega_ell_y: frequencies (along y-aixs) where the FourierData is taken
    :param omega_x_range: the min and max of the horizontal frequencies where
            the Fourier measurements are taken
    :param omega_y_range: the min and max of the vertical frequencies where
            the Fourier measurements are taken
    :param num_pixel_x: number of pixels horizontally
    :param num_pixel_y: number of pixels vertically
    :return:
    """
    num_pixel_x = (2 * np.floor(num_pixel_x / 2.) + 1).astype(int)  # ensure it is an odd number
    num_pixel_y = (2 * np.floor(num_pixel_y / 2.) + 1).astype(int)
    half_num_pixel_x = np.floor(num_pixel_x / 2.).astype(int)
    half_num_pixel_y = np.floor(num_pixel_y / 2.).astype(int)

    img_grid_ft = np.zeros((num_pixel_y, num_pixel_x), dtype=complex)
    # grid size
    step_sz_x = 2 * np.pi / tau_x
    step_sz_y = 2 * np.pi / tau_y

    # assuming the sampling frequencies are symmetric w.r.t. the origin
    num_samp = FourierData.size
    num_samp_half = np.floor(num_samp / 2.).astype(int)
    omega_ell_x_loc_half = np.round((omega_ell_x[0:num_samp_half]) /
                                    step_sz_x).astype(int) + half_num_pixel_x
    omega_ell_y_loc_half = np.round((omega_ell_y[0:num_samp_half]) /
                                    step_sz_y).astype(int) + half_num_pixel_y
    omega_ell_x_loc = np.concatenate((omega_ell_x_loc_half, -1 - omega_ell_x_loc_half))
    omega_ell_y_loc = np.concatenate((omega_ell_y_loc_half, -1 - omega_ell_y_loc_half))

    for loop in range(num_samp):
        img_grid_ft[omega_ell_y_loc[loop], omega_ell_x_loc[loop]] += FourierData[loop]
    if use_mkl_fft:
        img = np.real(sp_fft.fftshift(mkl_fft.ifft2(sp_fft.ifftshift(img_grid_ft))))
    else:
        img = np.real(sp_fft.fftshift(sp_fft.ifft2(sp_fft.fftshift(img_grid_ft))))
    img *= num_pixel_x * num_pixel_y
    return img, img_grid_ft


def mtx_freq2space(img_hat, omega_ell_x, omega_ell_y,
                   num_pixel_x, num_pixel_y, tau_x, tau_y):
    """
    build the linear operator that maps the Fourier transform
    to the discrete spatial domain image.
    :param img_hat: the Fourier transform that the linear operator applies to
    :param omega_ell_x: the horizontal frequency of the Fourier transform
    :param omega_ell_y: the vertical frequency of the Fourier transform
    :param num_pixel_x: number of pixels along the horizontal direction
            for the spatial domain image
    :param num_pixel_y: number of pixels along the vertical direction
            for the spatial domain image
    :param tau_x: the spatial domain image's support is from -0.5*tau to 0.5*tau
    :param tau_y: the spatial domain image's support is from -0.5*tau to 0.5*tau
    :return:
    """
    # ensure it is an odd number
    if num_pixel_x % 2 == 0:
        num_pixel_x += 1
    if num_pixel_y % 2 == 0:
        num_pixel_y += 1
    half_num_pixel_x = np.floor(num_pixel_x / 2.).astype(int)
    half_num_pixel_y = np.floor(num_pixel_y / 2.).astype(int)

    img_grid_ft = np.zeros((num_pixel_y, num_pixel_x), dtype=complex)
    # grid size
    step_sz_x = 2 * np.pi / tau_x
    step_sz_y = 2 * np.pi / tau_y
    # round frequencies to the nearest grid, which is defined by the number of pixels
    # assuming the sampling frequencies are symmetric w.r.t. the origin
    num_samp = img_hat.size
    num_samp_half = np.floor(num_samp / 2.).astype(int)
    omega_ell_x_loc_half = np.round((omega_ell_x[0:num_samp_half]) /
                                    step_sz_x).astype(int) + half_num_pixel_x
    omega_ell_y_loc_half = np.round((omega_ell_y[0:num_samp_half]) /
                                    step_sz_y).astype(int) + half_num_pixel_y
    omega_ell_x_loc = np.concatenate((omega_ell_x_loc_half, -1 - omega_ell_x_loc_half))
    omega_ell_y_loc = np.concatenate((omega_ell_y_loc_half, -1 - omega_ell_y_loc_half))
    # because there might be multiple references to the same entry several times, we use
    # for loop here instead of vectorized operation, which only applies the modification
    # to the index once.
    for loop in range(num_samp):
        img_grid_ft[omega_ell_y_loc[loop], omega_ell_x_loc[loop]] += img_hat[loop]
    if use_mkl_fft:
        img = sp_fft.fftshift(mkl_fft.ifft2(sp_fft.ifftshift(img_grid_ft)))
    else:
        img = sp_fft.fftshift(sp_fft.ifft2(sp_fft.fftshift(img_grid_ft)))
    return img


def mtx_space2freq(img, omega_ell_x, omega_ell_y,
                   num_pixel_x, num_pixel_y, tau_x, tau_y):
    """
    build the linear operator that maps the discrete spatial domain image
    to the Fourier transform.
    :param img: the spatial domain image that the linear operator applies to
    :param omega_ell_x: the horizontal frequency of the Fourier transform
    :param omega_ell_y: the vertical frequency of the Fourier transform
    :param num_pixel_x: number of pixels along the horizontal direction
            for the spatial domain image
    :param num_pixel_y: number of pixels along the vertical direction
            for the spatial domain image
    :param tau_x: the spatial domain image's support is from -0.5*tau to 0.5*tau
    :param tau_y: the spatial domain image's support is from -0.5*tau to 0.5*tau
    :return:
    """
    # ensure it is an odd number
    if num_pixel_x % 2 == 0:
        num_pixel_x += 1
    if num_pixel_y % 2 == 0:
        num_pixel_y += 1
    half_num_pixel_x = np.floor(num_pixel_x / 2.).astype(int)
    half_num_pixel_y = np.floor(num_pixel_y / 2.).astype(int)

    img_grid_ft = np.zeros((num_pixel_y, num_pixel_x), dtype=complex)
    # grid size
    step_sz_x = 2 * np.pi / tau_x
    step_sz_y = 2 * np.pi / tau_y
    # round frequencies to the nearest grid, which is defined by the number of pixels
    # assuming the sampling frequencies are symmetric w.r.t. the origin
    num_samp = omega_ell_x.size
    num_samp_half = np.floor(num_samp / 2.).astype(int)
    omega_ell_x_loc_half = np.round((omega_ell_x[0:num_samp_half]) /
                                    step_sz_x).astype(int) + half_num_pixel_x
    omega_ell_y_loc_half = np.round((omega_ell_y[0:num_samp_half]) /
                                    step_sz_y).astype(int) + half_num_pixel_y
    omega_ell_x_loc = np.concatenate((omega_ell_x_loc_half, -1 - omega_ell_x_loc_half))
    omega_ell_y_loc = np.concatenate((omega_ell_y_loc_half, -1 - omega_ell_y_loc_half))
    # because there might be multiple references to the same entry several times, we use
    # for loop here instead of vectorized operation, which only applies the modification
    # to the index once.
    if use_mkl_fft:
        img_hat_all = sp_fft.fftshift(mkl_fft.fft2(sp_fft.ifftshift(img)))
    else:
        img_hat_all = sp_fft.fftshift(sp_fft.fft2(sp_fft.ifftshift(img)))

    # extract the Fourier transforms specified by omega_ell_loc-s
    img_hat = img_hat_all[omega_ell_y_loc, omega_ell_x_loc]
    img_hat /= num_pixel_x * num_pixel_y
    return img_hat


# ============= for FISTA =============
def soft(x, th_level):
    """
    Soft-thresholding function
    :param x: input signal
    :param th_level: soft-thresholding level
    :return:
    """
    y = np.zeros_like(x)
    idx = np.abs(x) > th_level
    y[idx] = np.sign(x[idx]) * (np.abs(x[idx]) - th_level)
    return y


def power_method(A, in_sz, max_iter):
    """
    power method to find the largest eigen value of A
    :param A: the linear operator
    :param in_sz: input size
    :param max_iter: maximum iteration used
    :return:
    """
    # initilisation
    b = np.random.randn(*in_sz)
    for loop in range(max_iter):
        Ab = A(b)
        b = Ab / linalg.norm(Ab.flatten())

    Ab = A(b).flatten()
    b = b.flatten()
    return np.dot(b.T.conj(), Ab) / linalg.norm(b) ** 2


def fista(y, A, At, reg_weight, noise_eng, max_iter=100, update_reg=False, **kwargs):
    """
    The FISTA algorithm for the ell1 minimisation problem:
    min_x |y - Ax|^2 + reg * |x|_1
    :param y: the given measurements (here it is the Fourier transform at certain frequencies)
    :param A: the mapping from the sparse signal x to the measurements y
    :param At: the mapping from the measurements y to the sparse signal x
    :param reg_weight: regularisation weight for the ell1-norm of x
    :param noise_eng: noise energy, i.e., |y - Ax|^2
    :param max_iter: maximum number of FISTA iterations
    :param update_reg: whether to update the regularisation weight or not
    :param max_iter_reg: maximum number of iterations used to update the regularisation weight
    :return:
    """
    if not update_reg:
        max_iter_reg = 1
    else:
        max_iter_reg = kwargs['max_iter_reg']

    # initialise
    x = At(y)
    AtA = lambda input_arg: np.real(At(A(input_arg)))
    # Lipschitz constant for 2 * A^H Ax
    L = 1.01 * 2. * np.real(power_method(AtA, x.shape, 100))
    # print repr(L)  # for debug purposes
    for reg_loop in range(max_iter_reg):
        x = At(y)
        beta = x
        t_new = 1.
        for fista_loop in range(max_iter):
            x_old = x
            t_old = t_new
            # gradient step
            beta = beta - 2. / L * At(A(beta) - y)
            # soft-thresholding
            x = soft(beta, reg_weight / L)
            # update t and beta
            t_new = (1. + np.sqrt(1. + 4. * t_old ** 2)) / 2.
            beta = x + (t_old - 1.) / t_new * (x - x_old)

        reg_weight *= (noise_eng / linalg.norm(y - A(x)) ** 2)
    return x, reg_weight


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    Reference: http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    Modified by Hanjie Pan
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = np.double(maximum_filter(image, footprint=neighborhood) == image)

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line1 will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = np.double(binary_erosion(background, structure=neighborhood,
                                                 border_value=1))

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background
    peak_image = detected_peaks * image
    peak_locs = np.double(np.asarray(np.nonzero(detected_peaks)))
    return detected_peaks, peak_image, peak_locs
