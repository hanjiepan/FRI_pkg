# for the plot in the introduction
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
from alg_tools_2d import generate_dirty_img, mtx_space2freq, mtx_freq2space, fista, \
    recon_2d_dirac, plot_2d_dirac_spec, detect_peaks

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True
rcParams['text.latex.preamble'] = [r'\usepackage{bm}']

if __name__ == '__main__':
    num_realisation = 1000  # number of Gaussian noise realisations
    save_fig = True  # save figure or not
    fig_format = r'png'  # file type used to save the figure, e.g., pdf, png, etc.
    # number of Dirac
    K = 3
    K_est = 3  # estimated number of Diracs

    M = 12  # period of the spectrum along x-axis
    N = 12  # period of the spectrum along y-axis
    tau1 = 1
    tau2 = 1
    # amplitudes of the Dirac
    alpha_k = np.random.lognormal(mean=np.log(2), sigma=0.5, size=(K,))
    # locations of Diracs
    # multiplied by 0.6 is to prevent the Dirac from being located too close to the boundary
    xk = 0.9 * tau1 * np.random.rand()
    offset1 = 0.05 * np.sqrt(tau1 ** 2 + tau2 ** 2)
    offset2 = 0.06 * np.sqrt(tau1 ** 2 + tau2 ** 2)
    angle1 = np.pi * np.random.rand()
    angle2 = angle1 + np.pi * np.random.rand()
    xk = np.append(xk, np.array([xk + offset1 * np.cos(angle1), xk + offset2 * np.cos(angle2)]))
    xk -= 0.45 * tau1

    yk = 0.9 * tau2 * np.random.rand()
    yk = np.append(yk, np.array([yk + offset1 * np.sin(angle1), yk + offset2 * np.sin(angle2)]))
    yk -= 0.45 * tau2
    yk = yk[np.random.permutation(K)]

    # irregular frequency domain measurements
    L = 8000
    # cross-correlation is symmetric, so only need to specify half of the frequencies
    Lhalf = np.int(np.ceil(L / 2.))
    rand_num1 = (np.random.rand(Lhalf) + np.random.rand(Lhalf) + np.random.rand(Lhalf)) / 3.
    rand_num2 = (np.random.rand(Lhalf) + np.random.rand(Lhalf) + np.random.rand(Lhalf)) / 3.
    omega_ell_x_half = np.pi * (rand_num1 * (2 * M) - M)
    omega_ell_y_half = np.pi * (rand_num2 * (2 * N) - N)
    omega_ell_x = np.concatenate((omega_ell_x_half, -omega_ell_x_half))
    omega_ell_y = np.concatenate((omega_ell_y_half, -omega_ell_y_half))

    # save Dirac parameter
    '''
    time_stamp = datetime.datetime.now().strftime("%-d-%-m_%H_%M")
    file_name = './data/Dirac_Data_' + time_stamp + '.npz'
    np.savez(file_name, xk=xk, yk=yk, alpha_k=alpha_k, K=K,
            omega_ell_x=omega_ell_x, omega_ell_y=omega_ell_y)
    '''

    # load saved data
    time_stamp = r'8-2_19_47'
    stored_param = np.load('./data/Dirac_Data_' + time_stamp + '.npz')
    xk = stored_param['xk']
    yk = stored_param['yk']
    alpha_k = stored_param['alpha_k']
    K = stored_param['K'].tolist()
    omega_ell_x = stored_param['omega_ell_x']
    omega_ell_y = stored_param['omega_ell_y']

    print('time stamp: ' + time_stamp +
          '\n=======================================\n')

    L = omega_ell_x.size
    xk_grid, omega_grid_x = np.meshgrid(xk, omega_ell_x)
    yk_grid, omega_grid_y = np.meshgrid(yk, omega_ell_y)
    # Fourier measurements at frequencies omega_ell
    Ihat_omega_ell = np.dot(np.exp(-1j * omega_grid_x * xk_grid -
                                   1j * omega_grid_y * yk_grid), alpha_k)

    P = 5  # SNR in [dB]
    file_name_summary = './result/radio_ast_batch_res{}.npz'.format(num_realisation)
    if os.path.isfile(file_name_summary):
        result_all = np.load(file_name_summary)
        fri_recon_all = result_all['fri_recon_all']
        ell1_recon_all = result_all['ell1_recon_all']
    else:
        # initialisation
        fri_recon_all = np.zeros((num_realisation, 3 * K + 1))
        ell1_recon_all = np.zeros((num_realisation, 2 * K + 1))
        for realisations in range(num_realisation):
            # add noise
            # the added noise is Hermitian symmetric because the noise is added
            # to EM waves at each antenna. The Fourier transform is obtained via
            # cross-correlation. Hence, it will also be Hermitian symmetric.
            noise_half = np.random.randn(Lhalf) + 1j * np.random.randn(Lhalf)
            noise = np.concatenate((noise_half, np.conj(noise_half)))
            noise = noise / linalg.norm(noise) * linalg.norm(Ihat_omega_ell) * 10 ** (-P / 20.)
            Ihat_omega_ell_noisy = Ihat_omega_ell + noise

            # ==========================================================================
            # reconstruction algorithm to get denoised Fourier measurements on a uniform grid
            max_ini = 25
            stop_cri = 'max_iter'  # stopping criteria: 1) mse; or 2) max_iter
            noise_level = np.max([1e-10, linalg.norm(noise)])
            taus = np.array([tau1, tau2])
            omega_ell = np.column_stack((omega_ell_x, omega_ell_y))
            M_N = np.array([M, N])
            xk_recon, yk_recon, alpha_k_recon = recon_2d_dirac(Ihat_omega_ell_noisy,
                                                               K_est, tau1, tau2,
                                                               sympy.Rational(15, 12),
                                                               sympy.Rational(15, 12),
                                                               omega_ell, M_N[0], M_N[1],
                                                               noise_level, max_ini,
                                                               stop_cri, num_rotation=12)

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
            plt.figure(figsize=(3, 2.5), dpi=90)
            # subplot 1
            ax1 = plt.axes([0.18, 0.167, 0.619, 0.745])
            subplt22_13_1 = ax1.plot(xk, yk, label='Original Diracs')
            plt.setp(subplt22_13_1, linewidth=1.5,
                     color=[0, 0.447, 0.741], mec=[0, 0.447, 0.741], linestyle='None',
                     marker='^', markersize=8, markerfacecolor=[0, 0.447, 0.741])
            plt.axis('scaled')
            plt.xlim([-tau1 / 2., tau1 / 2.])
            plt.ylim([-tau2 / 2., tau2 / 2.])

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
            r_est_error_pow = np.int(np.floor(np.log10(r_est_error)))
            plt.title(r'Average $\mathbf{{r}}_{{\mbox{{\footnotesize '
                      r'error}}}}={0:.2f}\times10^{other}$'.format(r_est_error / 10 ** r_est_error_pow,
                                                                   other='{' + str(r_est_error_pow) + '}'),
                      fontsize=11)
            if save_fig:
                file_name_loc = (r'./result/TSP_intro_K_{0}_L_{1}_noise_{2}dB_locations.' +
                                 fig_format).format(repr(K), repr(L), repr(P))
                plt.savefig(file_name_loc, format=fig_format, dpi=300, transparent=True)

            # --------------------------------------------------------------------------

            file_name_spec = ('./result/TSP_intro_K_{0}_L_{1}_' +
                              'noise_{2}dB_spectrum').format(repr(K), repr(L), repr(P))
            plot_2d_dirac_spec(xk_recon, yk_recon, alpha_k_recon,
                               Ihat_omega_ell_noisy, Ihat_omega_ell,
                               omega_ell_x, omega_ell_y, M, N, P, L,
                               save_figure=save_fig, file_name=file_name_spec)

            # ==========================================================================
            # the ell1 minimisation result with FISTA
            num_pixel_x = 515
            num_pixel_y = 515
            A = lambda img: mtx_space2freq(img, omega_ell_x, omega_ell_y,
                                           num_pixel_x, num_pixel_y, tau1, tau2)
            At = lambda img_hat: np.real(mtx_freq2space(img_hat, omega_ell_x, omega_ell_y,
                                                        num_pixel_x, num_pixel_y, tau1, tau2))
            img_recon_ell1, reg_weight = fista(Ihat_omega_ell_noisy, A, At,
                                               3e-3, linalg.norm(noise) ** 2,
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

            print('Number of detected sources: {0}'.format(repr(peak_locs.shape[1])))

            # --------------------------------------------------------------------------
            # plot ell-1 reconstruction
            plt.figure(figsize=(3, 3), dpi=90)
            ax2 = plt.axes([0.2, 0.067, 0.75, 0.75])
            plt_ell1recon = ax2.imshow(np.real(img_recon_ell1) * (img_recon_ell1 > 0),
                                       origin='lower', cmap='Spectral_r')
            plt.xticks(np.linspace(num_pixel_x / 12., num_pixel_x - num_pixel_x / 12., num=5),
                       (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
            plt.yticks(np.linspace(num_pixel_y / 12., num_pixel_y - num_pixel_y / 12., num=5),
                       (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
            ax2c = plt.colorbar(plt_ell1recon, use_gridspec=False,
                                anchor=(-0.15, 0.5), shrink=0.8, spacing='proportional')
            ax2c.ax.tick_params(labelsize=8.5)
            ax2c.ax.yaxis.set_offset_position('left')
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
            if save_fig:
                file_name_ell1_recon = (r'./result/TSP_intro_K_{0}_L_{1}_' +
                                        r'noise_{2}dB_ell1_recon_{3}by{4}.' +
                                        fig_format).format(repr(K), repr(L), repr(P),
                                                           repr(num_pixel_y), repr(num_pixel_x))
                plt.savefig(file_name_ell1_recon, format=fig_format, dpi=300, transparent=True)

            # save batch run results
            np.savez(file_name_summary, fri_recon_all=fri_recon_all,
                     ell1_recon_all=ell1_recon_all)

    # ==========================================================================
    # dirty image that most astronomical processing tools start with
    num_pixel_x = 515
    num_pixel_y = 515

    noise_half = np.random.randn(Lhalf) + 1j * np.random.randn(Lhalf)
    noise = np.concatenate((noise_half, np.conj(noise_half)))
    noise = noise / linalg.norm(noise) * linalg.norm(Ihat_omega_ell) * 10 ** (-P / 20.)
    Ihat_omega_ell_noisy = Ihat_omega_ell + noise

    dirty_img, dirty_img_ft = generate_dirty_img(Ihat_omega_ell_noisy, omega_ell_x, omega_ell_y,
                                                 num_pixel_x, num_pixel_y, tau1, tau2)

    # --------------------------------------------------------------------------
    plt.figure(figsize=(3, 3), dpi=90)
    ax0 = plt.axes([0.2, 0.15, 0.75, 0.75])
    plt_spec = ax0.scatter(omega_ell_x, omega_ell_y, s=2, edgecolor='none',
                           vmin=np.abs(Ihat_omega_ell).min(),
                           vmax=np.abs(Ihat_omega_ell).max(),
                           c=np.abs(Ihat_omega_ell), cmap='Spectral')
    ax0c = plt.colorbar(plt_spec, use_gridspec=False,
                        anchor=(-0.1, 0.5), shrink=0.8, spacing='proportional')
    ax0c.formatter.set_powerlimits((0, 0))
    ax0c.ax.tick_params(labelsize=8.5)
    ax0c.ax.yaxis.set_offset_position('left')
    ax0c.update_ticks()
    plt.axis('scaled')
    plt.xlim([-np.pi * M, np.pi * M])
    plt.ylim([-np.pi * N, np.pi * N])
    plt.xlabel(r'$\omega_x$', fontsize=11)
    plt.ylabel(r'$\omega_y$', fontsize=11)
    plt.title(r'{\sf Irregular Fourier samples} $\hat{I}(\bm{\omega})$', fontsize=11)
    if save_fig:
        file_name_spec = (r'./result/TSP_intro_K_{0}_L_{1}_noise_{2}dB_spectrum.' +
                          fig_format).format(repr(K), repr(L), repr(P))
        plt.savefig(file_name_spec, format=fig_format, dpi=300, transparent=False)

    # --------------------------------------------------------------------------

    plt.figure(figsize=(3, 3), dpi=90)
    ax1 = plt.axes([0.2, 0.15, 0.75, 0.75])
    plt_dirtyimg = ax1.imshow(np.real(dirty_img), origin='lower', cmap='Spectral_r')
    plt.xticks(np.linspace(num_pixel_x / 12., num_pixel_x - num_pixel_x / 12., num=5),
               (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
    plt.yticks(np.linspace(num_pixel_y / 12., num_pixel_y - num_pixel_y / 12., num=5),
               (r'$-0.4$', r'$-0.2$', r'$0.0$', r'$0.2$', r'$0.4$'))
    ax1c = plt.colorbar(plt_dirtyimg, use_gridspec=False,
                        anchor=(-0.1, 0.5), shrink=0.8, spacing='proportional')
    ax1c.formatter.set_powerlimits((0, 0))
    ax1c.ax.tick_params(labelsize=8.5)
    ax1c.ax.yaxis.set_offset_position('left')
    ax1c.update_ticks()
    plt.xlabel(r'horizontal position $x$', fontsize=11)
    plt.ylabel(r'vertical position $y$', fontsize=11)
    ax1.xaxis.set_label_coords(0.5, -0.11)
    ax1.yaxis.set_label_coords(-0.19, 0.5)
    plt.title(r'Dirty Image ' +
              r'(size: ${0}\times{1}$)'.format(repr(num_pixel_y), repr(num_pixel_y)),
              fontsize=11)
    if save_fig:
        file_name_dirty_img = (r'./result/TSP_intro_K_{0}_L_{1}_' +
                               r'noise_{2}dB_dirty_img_{3}by{4}.' +
                               fig_format).format(repr(K), repr(L), repr(P),
                                                  repr(num_pixel_y), repr(num_pixel_x))
        plt.savefig(file_name_dirty_img, format=fig_format, dpi=300, transparent=False)

    # ==========================================================================

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
    cbar = plt.colorbar(cax, anchor=(-0.1, 0.5), shrink=0.8, location='right',
                        ticks=[[z_loop.min(), (z_loop.min() + z_loop.max()) / 2.,
                                z_loop.max()]])
    cbar.ax.set_yticklabels([r'low', r'medium', r'high'], fontsize=8, rotation=270)
    cbar.ax.xaxis.set_tick_params(pad=0.3)
    cbar.ax.yaxis.set_offset_position('left')
    plt.axis('scaled')
    plt.xlim([-0.5 * tau1, 0.5 * tau1])
    plt.ylim([-0.5 * tau2, 0.5 * tau2])
    plt.xlabel(r'horizontal position $x$', fontsize=11)
    ax3.xaxis.set_label_coords(0.5, -0.11)
    plt.title(r'Probability Density', fontsize=11)

    if save_fig:
        file_name_aggregated = (r'./result/TSP_intro_K_{0}_L_{1}_' +
                                r'noise_{2}dB_aggregated.' +
                                fig_format).format(repr(K), repr(L), repr(P))
        plt.savefig(file_name_aggregated, format=fig_format, dpi=300, transparent=False)

    # --------------------------------------------------------------------------

    # zoom-in plot for both the dirty image and the statistical plot
    zoom_box_x = [0.14 * tau1, 0.3 * tau1]
    zoom_box_y = [-0.1 * tau2, 0.06 * tau2]
    plt.figure(figsize=(3, 3), dpi=90)
    ax4 = plt.axes([0.2, 0.15, 0.75, 0.75])
    for k in range(K):
        data_xy = np.vstack((np.reshape(fri_recon_all[:endpoint + 1, k],
                                        (1, -1), order='F'),
                             np.reshape(fri_recon_all[:endpoint + 1, k + K],
                                        (1, -1), order='F')
                             ))
        z_loop = stats.gaussian_kde(data_xy, 'silverman')(data_xy)
        idx_loop = z_loop.argsort()
        cax = ax4.scatter(fri_recon_all[:endpoint + 1, k][idx_loop],
                          fri_recon_all[:endpoint + 1, k + K][idx_loop],
                          s=0.25, edgecolor='none', c=z_loop[idx_loop], cmap='Spectral_r')
        if k == 0:
            ax4.hold(True)

    cbar = plt.colorbar(cax, anchor=(-0.1, 0.5), shrink=0.8, location='right',
                        ticks=[[z_loop.min(), (z_loop.min() + z_loop.max()) / 2.,
                                z_loop.max()]])
    cbar.ax.set_yticklabels([r'low', r'medium', r'high'], fontsize=8, rotation=270)
    cbar.ax.xaxis.set_tick_params(pad=0.3)
    cbar.ax.yaxis.set_offset_position('left')
    # cbar.update_ticks()
    plt.axis('scaled')
    plt.xlim([0.14 * tau1, 0.3 * tau1])
    plt.ylim([-0.1 * tau2, 0.06 * tau2])
    ax4.xaxis.major.locator.set_params(nbins=5)
    ax4.yaxis.major.locator.set_params(nbins=5)
    plt.xlabel(r'horizontal position $x$', fontsize=11)
    ax4.xaxis.set_label_coords(0.5, -0.11)
    if save_fig:
        file_name_aggregated_zoom = ('./result/TSP_intro_K_{0}_L_{1}_' +
                                     'noise_{2}dB_aggregated_zoom.' +
                                     fig_format).format(repr(K), repr(L), repr(P))
        plt.savefig(file_name_aggregated_zoom, format=fig_format, dpi=300, transparent=False)

    # --------------------------------------------------------------------------

    # zoom in of the dirty image
    plt.figure(figsize=(3, 3), dpi=90)
    ax5 = plt.axes([0.2, 0.15, 0.75, 0.75])
    plt_dirtyimg_zoom = ax5.imshow(np.real(dirty_img), origin='lower', cmap='Spectral_r')
    plt.xticks([0.65 * (num_pixel_x - 1), 0.7 * (num_pixel_x - 1),
                0.75 * (num_pixel_x - 1), 0.8 * (num_pixel_x - 1)],
               (r'$0.15$', r'$0.20$', r'$0.25$', r'$0.30$'))
    plt.yticks([0.55 * (num_pixel_y - 1), 0.5 * (num_pixel_y - 1),
                0.45 * (num_pixel_y - 1), 0.4 * (num_pixel_y - 1)],
               (r'$0.05$', r'$0.00$', r'$-0.05$', r'$-0.10$'))
    ax5c = plt.colorbar(plt_dirtyimg_zoom, use_gridspec=False,
                        anchor=(-0.1, 0.5), shrink=0.8, spacing='proportional')
    ax5c.formatter.set_powerlimits((0, 0))
    ax5c.ax.tick_params(labelsize=8.5)
    ax5c.ax.yaxis.set_offset_position('left')
    ax5c.update_ticks()

    plt.xlabel(r'horizontal position $x$', fontsize=11)
    plt.ylabel(r'vertical position $y$', fontsize=11)
    ax5.xaxis.set_label_coords(0.5, -0.11)
    ax5.yaxis.set_label_coords(-0.21, 0.5)
    plt.xlim([0.64 * (num_pixel_x - 1), 0.8 * (num_pixel_x - 1)])
    plt.ylim([0.4 * (num_pixel_y - 1), 0.56 * (num_pixel_y - 1)])

    if save_fig:
        file_name_dirty_img_zoom = ('./result/TSP_intro_K_{0}_L_{1}_noise_' +
                                    '{2}dB_dirty_img_{3}by{4}_zoom.' +
                                    fig_format).format(repr(K), repr(L), repr(P),
                                                       repr(num_pixel_y), repr(num_pixel_x))
        plt.savefig(file_name_dirty_img_zoom, format=fig_format, dpi=300, transparent=False)
    plt.show()

    # --------------------------------------
    plt.close('all')
