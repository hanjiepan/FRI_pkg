from __future__ import division
import datetime
import os
import numpy as np
from scipy import linalg, stats
import sympy
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
else:
    matplotlib.use('Qt5Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from alg_tools_1d import distance
from alg_tools_2d import mtx_space2freq, mtx_freq2space, fista, \
    recon_2d_dirac, plot_2d_dirac_loc, plot_2d_dirac_spec, detect_peaks

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True

if __name__ == '__main__':
    num_realisation = 1000
    save_fig = True  # save figure or not
    fig_format = r'png'  # file type used to save the figure, e.g., pdf, png, etc.
    # number of Dirac
    K = 5
    K_est = 5  # estimated number of Diracs
    # M * tau1 and N * tau2 are odd numbers
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
    '''
    # generate random Fourier domain sampling locations, which have denser distributions in low frequencies
    # cross-correlation is symmetric, so only need to specify half of the frequencies
    Lhalf = np.int(np.ceil(L / 2.))
    rand_num1 = (np.random.rand(Lhalf) + np.random.rand(Lhalf) + np.random.rand(Lhalf)) / 3.
    rand_num2 = (np.random.rand(Lhalf) + np.random.rand(Lhalf) + np.random.rand(Lhalf)) / 3.
    omega_ell_x_half = np.pi * (rand_num1 * (2 * M) - M)
    omega_ell_y_half = np.pi * (rand_num2 * (2 * N) - N)
    omega_ell_x = np.concatenate((omega_ell_x_half, -omega_ell_x_half))
    omega_ell_y = np.concatenate((omega_ell_y_half, -omega_ell_y_half))
    '''

    # load saved data
    time_stamp = r'4-2_02_28'
    stored_param = np.load(r'./data/Dirac_Data_' + time_stamp + r'.npz')
    xk = stored_param['xk']
    yk = stored_param['yk']
    alpha_k = stored_param['alpha_k']
    K = stored_param['K'].tolist()
    omega_ell_x = stored_param['omega_ell_x']
    omega_ell_y = stored_param['omega_ell_y']

    print(r'time stamp: ' + time_stamp +
          '\n=======================================\n')

    L = omega_ell_x.size
    Lhalf = np.int(np.ceil(L / 2.))
    xk_grid, omega_grid_x = np.meshgrid(xk, omega_ell_x)
    yk_grid, omega_grid_y = np.meshgrid(yk, omega_ell_y)
    # Fourier measurements at frequencies omega_ell
    Ihat_omega_ell = np.dot(np.exp(- 1j * omega_grid_x * xk_grid
                                   - 1j * omega_grid_y * yk_grid), alpha_k)

    # add noise
    P = 5  # SNR in [dB]
    file_name_summary = r'./result/radio_ast_eg3_batch_res' + repr(num_realisation) + r'.npz'
    if os.path.isfile(file_name_summary):
        result_all = np.load(file_name_summary)
        fri_recon_all = result_all['fri_recon_all']
        ell1_recon_all = result_all['ell1_recon_all']
    else:
        # initialisation
        fri_recon_all = np.zeros((num_realisation, 3 * K + 1))
        ell1_recon_all = np.zeros((num_realisation, 2 * K + 1))

        for realisations in range(num_realisation):
            # the added noise is Hermitian symmetric because the noise is added
            # to EM waves at each antenna. The Fourier transform is obtained via
            # cross-correlation. Hence, it will also be Hermitian symmetric.
            noise_half = np.random.randn(Lhalf) + 1j * np.random.randn(Lhalf)
            noise = np.concatenate((noise_half, np.conj(noise_half)))
            noise = noise / linalg.norm(noise) * linalg.norm(Ihat_omega_ell) * 10 ** (-P / 20.)
            Ihat_omega_ell_noisy = Ihat_omega_ell + noise

            # reconstruction algorithm to get denoised Fourier measurements on a uniform grid
            max_ini = 25
            stop_cri = 'max_iter'  # stopping criteria: 1) mse; or 2) max_iter
            noise_level = np.max([1e-10, linalg.norm(noise)])
            taus = np.array([tau1, tau2])
            omega_ell = np.column_stack((omega_ell_x, omega_ell_y))
            xk_recon, yk_recon, alpha_k_recon = \
                recon_2d_dirac(Ihat_omega_ell_noisy, K_est, tau1, tau2,
                               sympy.Rational(15, 12), sympy.Rational(15, 12),
                               omega_ell, M, N, noise_level,
                               max_ini, stop_cri, num_rotation=12)
            # calculate reconstruction error
            r_est_error, index = distance(xk + 1j * yk, xk_recon + 1j * yk_recon)
            xk = xk[index[:, 0]]
            yk = yk[index[:, 0]]
            alpha_k = alpha_k[index[:, 0]]
            xk_recon = xk_recon[index[:, 1]]
            yk_recon = yk_recon[index[:, 1]]
            alpha_k_recon = np.real(alpha_k_recon[index[:, 1]])
            # order the results for the ease of comparison
            ind_order = np.argsort(xk)
            fri_recon_all[realisations, :] = np.hstack((np.reshape(xk_recon[ind_order],
                                                                   (1, K), order='F'),
                                                        np.reshape(yk_recon[ind_order],
                                                                   (1, K), order='F'),
                                                        np.reshape(alpha_k_recon[ind_order],
                                                                   (1, K), order='F'),
                                                        np.reshape(r_est_error, (1, 1), order='F')))
            print('Position estimation error: {0:.2e}\n').format(r_est_error)

            # --------------------------------------------------------------------------
            # plot results
            plt.close('all')
            file_name_loc = (r'./result/TSP_eg4_K_{0}_L_{1}_' +
                             r'noise_{2}dB_locations.' +
                             fig_format).format(repr(K), repr(L), repr(P))
            plot_2d_dirac_loc(xk_recon, yk_recon, alpha_k_recon, xk, yk, alpha_k, K, L, P, tau1, tau2,
                              save_figure=True, fig_format=fig_format, file_name=file_name_loc)
            # --------------------------------------------------------------------------
            file_name_spec = (r'./result/TSP_eg4_K_{0}_L_{1}_' +
                              r'noise_{2}dB_spectrum').format(repr(K), repr(L), repr(P))
            plot_2d_dirac_spec(xk_recon, yk_recon, alpha_k_recon, Ihat_omega_ell_noisy, Ihat_omega_ell,
                               omega_ell_x, omega_ell_y, M, N, P, L,
                               save_figure=True, fig_format=fig_format,
                               file_name=file_name_spec)

            # ==========================================================================
            # the ell1 minimisation result with FISTA
            num_pixel_x = 515
            num_pixel_y = 515
            A = lambda img: mtx_space2freq(img, omega_ell_x, omega_ell_y,
                                           num_pixel_x, num_pixel_y, tau1, tau2)
            At = lambda img_hat: np.real(mtx_freq2space(img_hat, omega_ell_x, omega_ell_y,
                                                        num_pixel_x, num_pixel_y, tau1, tau2))
            img_recon_ell1, reg_weight = fista(Ihat_omega_ell_noisy, A, At,
                                               4e-3, linalg.norm(noise) ** 2,
                                               max_iter=200, max_iter_reg=200)
            # --------------------------------------------------------------------------
            # detect local maximum points
            peak_locs = detect_peaks(img_recon_ell1 * (img_recon_ell1 > 0))[2]
            xk_recon_ell1 = tau1 * peak_locs[1, :] / num_pixel_x - 0.5 * tau1
            yk_recon_ell1 = tau2 * peak_locs[0, :] / num_pixel_y - 0.5 * tau2
            if peak_locs.shape[1] == 0:
                r_est_error_ell1, index_ell1 = distance(xk + 1j * yk,
                                                        np.zeros(K, dtype=complex))
            else:
                r_est_error_ell1, index_ell1 = distance(xk + 1j * yk,
                                                        xk_recon_ell1 + 1j * yk_recon_ell1)

            xk = xk[index_ell1[:, 0]]
            yk = yk[index_ell1[:, 0]]
            alpha_k = alpha_k[index_ell1[:, 0]]
            xk_recon_ell1 = xk_recon_ell1[index_ell1[:, 1]]
            yk_recon_ell1 = yk_recon_ell1[index_ell1[:, 1]]
            # order the results for the ease of comparison
            ind_order_ell1 = np.argsort(xk)
            ell1_recon_all[realisations, :] = np.hstack((np.reshape(xk_recon_ell1[ind_order_ell1],
                                                                    (1, K), order='F'),
                                                         np.reshape(yk_recon_ell1[ind_order_ell1],
                                                                    (1, K), order='F'),
                                                         np.reshape(r_est_error_ell1,
                                                                    (1, 1), order='F')))

            print('Number of detected sources: {0}\n'.format(repr(peak_locs.shape[1])))

            # --------------------------------------------------------------------------
            # plot ell-1 reconstruction
            plt.figure(figsize=(3, 3), dpi=90)
            ax2 = plt.axes([0.2, 0.067, 0.75, 0.75])
            plt_ell1recon = ax2.imshow(np.real(img_recon_ell1) * (img_recon_ell1 > 0), origin='lower',
                                       cmap='Spectral_r')
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
            plt.title(r'Average $\mathbf{{r}}_{{\mbox{{\footnotesize error}}}}'
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

            # save batch run results
            np.savez(file_name_summary, fri_recon_all=fri_recon_all,
                     ell1_recon_all=ell1_recon_all)

    # plot the aggregated results
    plt.figure(figsize=(3, 3), dpi=90)
    ax3 = plt.axes([0.2, 0.15, 0.75, 0.75])
    endpoint = fri_recon_all[:, 0].nonzero()[0][-1]
    for k in range(K):
        data_xy = np.vstack((np.reshape(fri_recon_all[:endpoint + 1, k],
                                        (1, -1), order='F'),
                             np.reshape(fri_recon_all[:endpoint + 1, k + K],
                                        (1, -1), order='F')
                             ))
        z_loop = stats.gaussian_kde(data_xy, 'silverman')(data_xy)

        idx_loop = z_loop.argsort()
        cax = ax3.scatter(fri_recon_all[:endpoint + 1, k][idx_loop],
                          fri_recon_all[:endpoint + 1, k + K][idx_loop],
                          s=0.25, edgecolor='none', c=z_loop[idx_loop], cmap='Spectral_r')
        if k == 0:
            ax3.hold(True)
    cbar = plt.colorbar(cax, shrink=0.8, anchor=(-0.1, 0.5), location='right',
                        ticks=[[z_loop.min(), (z_loop.min() + z_loop.max()) / 2.,
                                z_loop.max()]])
    cbar.ax.set_yticklabels(['low', 'medium', 'high'], fontsize=8, rotation=270)
    cbar.ax.xaxis.set_tick_params(pad=0.3)

    plt.axis('scaled')
    plt.xlim([-0.5 * tau1, 0.5 * tau1])
    plt.ylim([-0.5 * tau2, 0.5 * tau2])
    plt.xlabel(r'horizontal position $x$', fontsize=11)
    plt.ylabel(r'vertical position $y$', fontsize=11)
    ax3.xaxis.set_label_coords(0.5, -0.11)
    ax3.yaxis.set_label_coords(-0.19, 0.5)
    plt.title(r'Probability Density', fontsize=11)

    if save_fig:
        file_name_aggregated = (r'./result/TSP_intro_K_{0}_L_{1}_' +
                                r'noise_{2}dB_aggregated.' +
                                fig_format).format(repr(K), repr(L), repr(P))
        plt.savefig(file_name_aggregated, format=fig_format, dpi=300, transparent=True)
    plt.show()
