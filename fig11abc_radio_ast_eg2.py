from __future__ import division
import datetime
import os
import numpy as np
from numpy.random import RandomState
from scipy import linalg
import time
import sympy
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
else:
    matplotlib.use('Qt5Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from alg_tools_1d import distance
from alg_tools_2d import generate_dirty_img, mtx_space2freq, mtx_freq2space, fista, \
    recon_2d_dirac, plot_2d_dirac_loc, plot_2d_dirac_spec, detect_peaks

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True

if __name__ == '__main__':
    save_fig = False  # save figure or not
    fig_format = r'png'  # file type used to save the figure, e.g., pdf, png, etc.
    # number of Dirac
    K = 5
    K_est = 5  # estimated number of Diracs

    M = 12  # period of the spectrum along x-axis: M * tau1 must be an ODD number
    N = 12  # period of the spectrum along y-axis: N * tau2 must be an ODD number
    tau1 = 1
    tau2 = 1
    # amplitudes of the Dirac
    alpha_k = np.random.lognormal(mean=np.log(2), sigma=0.5, size=(K,))
    # locations of Diracs
    a1 = 1 / M
    a2 = 1 / N
    uk_x = np.random.exponential(1. / (K - 1), (1, K - 1))
    # multiplied by 0.9 is to prevent the Dirac from being located too close to the boundary
    xk = 0.9 * np.cumsum(a1 + (1 - (K - 1) * a1) * (1. - 0.1 * np.random.rand(1, 1)) / np.sum(uk_x) * uk_x)
    offset = 0.06 * np.sqrt(tau1 ** 2 + tau2 ** 2)
    angle = 2 * np.pi * np.random.rand()
    xk = np.append(xk, xk[np.int(K / 2)] - offset * np.cos(angle))
    xk -= 0.45 * tau1

    uk_y = np.random.exponential(1. / (K - 1), (1, K - 1))
    yk = 0.9 * np.cumsum(a2 + (1 - (K - 1) * a2) * (1. - 0.1 * np.random.rand(1, 1)) / np.sum(uk_y) * uk_y)
    yk -= 0.45 * tau2
    yk = yk[np.random.permutation(K - 1)]
    yk = np.append(yk, yk[np.int(K / 2)] - offset * np.sin(angle))

    # irregular frequency domain measurements
    L = 8500
    # cross-correlation is symmetric, so only need to specify half of the frequencies
    Lhalf = np.int(np.ceil(L / 2.))

    '''
    rand_num1 = (np.random.rand(Lhalf) + np.random.rand(Lhalf) + np.random.rand(Lhalf)) / 3.
    rand_num2 = (np.random.rand(Lhalf) + np.random.rand(Lhalf) + np.random.rand(Lhalf)) / 3.
    omega_ell_x_half = np.pi * (rand_num1 * (2 * M) - M)
    omega_ell_y_half = np.pi * (rand_num2 * (2 * N) - N)
    omega_ell_x = np.concatenate((omega_ell_x_half, -omega_ell_x_half))
    omega_ell_y = np.concatenate((omega_ell_y_half, -omega_ell_y_half))

    # save Dirac parameter
    time_stamp = datetime.datetime.now().strftime("%-d-%-m_%H_%M")
    file_name = r'./data/Dirac_Data_' + time_stamp + r'.npz'
    np.savez(file_name, xk=xk, yk=yk, alpha_k=alpha_k, K=K,
             omega_ell_x=omega_ell_x, omega_ell_y=omega_ell_y,
             time_stamp=time_stamp)
    '''

    # load saved data
    time_stamp = r'4-2_02_28'
    stored_param = np.load(r'./data/Dirac_Data_' + time_stamp + '.npz')
    xk = stored_param['xk']
    yk = stored_param['yk']
    alpha_k = stored_param['alpha_k']
    K = stored_param['K'].tolist()
    omega_ell_x = stored_param['omega_ell_x']
    omega_ell_y = stored_param['omega_ell_y']

    print(r'time stamp: ' + time_stamp +
          '\n=======================================\n')

    L = omega_ell_x.size
    xk_grid, omega_grid_x = np.meshgrid(xk, omega_ell_x)
    yk_grid, omega_grid_y = np.meshgrid(yk, omega_ell_y)
    # Fourier measurements at frequencies omega_ell
    Ihat_omega_ell = np.dot(np.exp(- 1j * omega_grid_x * xk_grid
                                   - 1j * omega_grid_y * yk_grid), alpha_k)
    # add noise
    P = 5  # SNR in [dB]
    # the added noise is Hermitian symmetric because the noise is added
    # to EM waves at each antenna. The Fourier transform is obtained via
    # cross-correlation. Hence, it will also be Hermitian symmetric.
    noise_half = np.random.randn(Lhalf) + 1j * np.random.randn(Lhalf)
    noise = np.concatenate((noise_half, np.conj(noise_half)))
    noise = noise / linalg.norm(noise) * linalg.norm(Ihat_omega_ell) * 10 ** (-P / 20.)
    Ihat_omega_ell_noisy = Ihat_omega_ell + noise

    # dirty image that most astronomical processing tools start with
    num_pixel_x = 515
    num_pixel_y = 515

    dirty_img, dirty_img_ft = generate_dirty_img(Ihat_omega_ell_noisy, omega_ell_x, omega_ell_y,
                                                 num_pixel_x, num_pixel_y, tau1, tau2)
    plt.figure(figsize=(3, 3), dpi=90)
    ax1 = plt.axes([0.2, 0.067, 0.75, 0.75])
    plt_dirtyimg = ax1.imshow(np.real(dirty_img), origin='lower', cmap='Spectral_r')
    plt.xticks(np.linspace(num_pixel_x / 12., num_pixel_x - num_pixel_x / 12., num=5),
               (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
    plt.yticks(np.linspace(num_pixel_y / 12., num_pixel_y - num_pixel_y / 12., num=5),
               (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
    ax1c = plt.colorbar(plt_dirtyimg, use_gridspec=False,
                        anchor=(-0.15, 0.5), shrink=0.8, spacing='proportional')
    ax1c.ax.tick_params(labelsize=8.5)
    plt.xlabel(r'horizontal position $x$', fontsize=12)
    plt.ylabel(r'vertical position $y$', fontsize=12)
    ax1.xaxis.set_label_coords(0.5, -0.11)
    ax1.yaxis.set_label_coords(-0.19, 0.5)
    plt.title(r'Dirty Image ' +
              r'(size: ${0}\times{1}$)'.format(repr(num_pixel_y), repr(num_pixel_y)),
              fontsize=11)
    file_name_dirty_img = (r'./result/TSP_eg4_K_{0}_L_{1}_' +
                           r'noise_{2}dB_dirty_img_{3}by{4}.' +
                           fig_format).format(repr(K), repr(L), repr(P),
                                              repr(num_pixel_y), repr(num_pixel_x))
    plt.savefig(file_name_dirty_img, format=fig_format, dpi=300, transparent=True)

    # reconstruction algorithm to get denoised Fourier measurements on a uniform grid
    max_ini = 25
    stop_cri = 'max_iter'  # stopping criteria: 1) mse; or 2) max_iter
    noise_level = np.max([1e-10, linalg.norm(noise)])
    taus = np.array([tau1, tau2])
    omega_ell = np.column_stack((omega_ell_x, omega_ell_y))

    tic = time.time()
    num_rotation = 12
    xk_recon, yk_recon, alpha_k_recon = \
        recon_2d_dirac(Ihat_omega_ell_noisy, K_est, tau1, tau2,
                       sympy.Rational(15, 12), sympy.Rational(15, 12),
                       omega_ell, M, N, noise_level,
                       max_ini, stop_cri, num_rotation)
    toc = time.time()
    print('Average time: {0:.2f}[sec]'.format((toc - tic) / num_rotation))

    # calculate reconstruction error
    r_est_error = distance(xk + 1j * yk, xk_recon + 1j * yk_recon)[0]
    print('Position estimation error: {0:.2e}\n'.format(r_est_error))

    # plot results
    file_name_loc = (r'./result/TSP_eg4_K_{0}_L_{1}_' +
                     r'noise_{2}dB_locations.' +
                     fig_format).format(repr(K), repr(L), repr(P))
    plot_2d_dirac_loc(xk_recon, yk_recon, alpha_k_recon, xk, yk, alpha_k, K, L, P, tau1, tau2,
                      save_figure=True, fig_format=fig_format, file_name=file_name_loc)
    file_name_spec = (r'./result/TSP_eg4_K_{0}_L_{1}_' +
                      r'noise_{2}dB_spectrum').format(repr(K), repr(L), repr(P))
    plot_2d_dirac_spec(xk_recon, yk_recon, alpha_k_recon, Ihat_omega_ell_noisy, Ihat_omega_ell,
                       omega_ell_x, omega_ell_y, M, N, P, L,
                       save_figure=True, fig_format=fig_format,
                       file_name=file_name_spec,
                       show_dirty_img=True, dirty_img=dirty_img)

    # the ell1 minimisation result with FISTA
    A = lambda img: mtx_space2freq(img, omega_ell_x, omega_ell_y,
                                   num_pixel_x, num_pixel_y, tau1, tau2)
    At = lambda img_hat: np.real(mtx_freq2space(img_hat, omega_ell_x, omega_ell_y,
                                                num_pixel_x, num_pixel_y, tau1, tau2))
    img_recon_ell1, reg_weight = fista(Ihat_omega_ell_noisy, A, At,
                                       4e-3, linalg.norm(noise) ** 2,
                                       max_iter=200, max_iter_reg=200)
    peak_locs = detect_peaks(img_recon_ell1 * (img_recon_ell1 > 0))[2]
    xk_recon_ell1 = tau1 * peak_locs[1, :] / num_pixel_x - 0.5 * tau1
    yk_recon_ell1 = tau2 * peak_locs[0, :] / num_pixel_y - 0.5 * tau2
    if peak_locs.shape[1] == 0:
        r_est_error_ell1 = distance(xk + 1j * yk, np.zeros(K, dtype=complex))[0]
    else:
        r_est_error_ell1 = distance(xk + 1j * yk, xk_recon_ell1 + 1j * yk_recon_ell1)[0]
    print('Number of detected sources: {0}\n'.format(repr(peak_locs.shape[1])))

    plt.figure(figsize=(3, 3), dpi=90)
    ax2 = plt.axes([0.2, 0.067, 0.75, 0.75])
    plt_ell1recon = ax2.imshow(np.real(img_recon_ell1) * (img_recon_ell1 > 0), origin='lower', cmap='Spectral_r')
    plt.xticks(np.linspace(num_pixel_x / 12., num_pixel_x - num_pixel_x / 12., num=5),
               (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
    plt.yticks(np.linspace(num_pixel_y / 12., num_pixel_y - num_pixel_y / 12., num=5),
               (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
    ax2c = plt.colorbar(plt_ell1recon, use_gridspec=False,
                        anchor=(-0.15, 0.5), shrink=0.8, spacing='proportional')
    ax2c.ax.tick_params(labelsize=8.5)
    plt.xlabel(r'horizontal position $x$', fontsize=12)
    plt.ylabel(r'vertical position $y$', fontsize=12)
    ax2.xaxis.set_label_coords(0.5, -0.11)
    ax2.yaxis.set_label_coords(-0.19, 0.5)
    r_est_error_ell1_pow = np.int(np.floor(np.log10(r_est_error_ell1)))
    plt.title(r'$\mathbf{{r}}_{{\mbox{{\footnotesize error}}}}'
              r'={0:.2f}\times10^{other}$'.format(r_est_error_ell1 /
                                                  10 ** r_est_error_ell1_pow,
                                                  other='{' + str(r_est_error_ell1_pow) + '}'),
              fontsize=11)

    file_name_ell1_recon = (r'./result/TSP_eg4_K_{0}_L_{1}_' +
                            r'noise_{2}dB_ell1_recon_{3}by{4}.' +
                            fig_format).format(repr(K), repr(L), repr(P),
                                               repr(num_pixel_y), repr(num_pixel_x))
    plt.savefig(file_name_ell1_recon, format=fig_format, dpi=300, transparent=True)
    plt.show()
