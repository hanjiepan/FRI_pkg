from __future__ import division
import datetime
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from alg_tools_1d import distance, build_G_fourier, dirac_recon_irreg_fourier, dirac_recon_time

if __name__ == '__main__':
    # various experiment settings
    save_fig = True  # save figure or not
    fig_format = r'png'  # file type used to save the figure, e.g., pdf, png, etc.
    interp_kernel = 'dirichlet'  # interpolation kernel used: 1) dirichlet; 2) triangular; 3) cubic; 4) keys
    periodic_spectrum = False  # whether the spetrum is periodic or not

    # check and correct interpolation kernel if the spectrum is periodic
    # should use dirichlet kernel in this case
    if periodic_spectrum:
        interp_kernel = 'dirichlet'

    K = 5  # number of Diracs
    M = 21  # period of the spectrum: M * tau must be an ODD number
    tau = 1.  # time support of the signal is from -0.5 * tau to 0.5 * tau

    '''
    # generate random values for Dirac amplitudes and locations
    # amplitudes of the Diracs
    ak = np.sign(np.random.randn(K)) * (1 + (np.random.rand(K) - 0.5) / 1.)
    # locations of the Diracs
    if periodic_spectrum:
        delta_t = 1. / M
        tk = np.sort(
            np.random.permutation(np.uint(np.floor(0.5 * tau / delta_t) * 2))[0:K] *
            delta_t) - np.floor(0.5 * tau / delta_t) * delta_t
    else:
        if K == 1:
            tk = np.random.rand()
        else:
            a = 4. / M
            uk = np.random.exponential(scale=1. / K, size=(K - 1, 1))
            tk = np.cumsum(a + (1. - K * a) * (1 - 0.1 * np.random.rand()) / uk.sum() * uk)
            tk = np.sort(np.hstack((np.random.rand() * tk[0] / 2., tk)) + (1 - tk[-1]) / 2.) * tau - 0.5 * tau

    # save Dirac parameter
    time_stamp = datetime.datetime.now().strftime("%d-%m_%H_%M")
    if periodic_spectrum:
        file_name = './data/freq_Dirac_Data_p' + time_stamp + '.npz'
    else:
        file_name = './data/freq_Dirac_Data_ap' + time_stamp + '.npz'
    np.savez(file_name, tk=tk, ak=ak, K=K, M=M, time_stamp=time_stamp)
    '''

    # load saved data
    if periodic_spectrum:
        time_stamp = r'20-12_11_48'  # for the periodic case
    else:
        time_stamp = r'20-12_11_55'  # for the aperiodic case
    if periodic_spectrum:
        stored_param = np.load('./data/freq_Dirac_Data_p' + time_stamp + '.npz')
    else:
        stored_param = np.load('./data/freq_Dirac_Data_ap' + time_stamp + '.npz')
    tk = stored_param['tk']
    ak = stored_param['ak']
    K = stored_param['K'].tolist()
    M = stored_param['M'].tolist()

    print(r'time stamp: ' + time_stamp +
          '\n=======================================\n')

    # number of random Fourier domain measurements (over sample in the noisy case)
    L = 2 * M + 1
    # generate random frequencies to take samples
    omega_ell = np.linspace(-M * np.pi, M * np.pi, L)
    omega_step_sz = omega_ell[1] - omega_ell[0]
    # Fourier transform of the Diracs
    tk_grid, omega_grid = np.meshgrid(tk, omega_ell)
    Xomega_ell = np.dot(np.exp(-1j * omega_grid * tk_grid), ak)

    # add noise
    P = 5#np.inf
    noise = np.random.randn(L) + 1j * np.random.randn(L)
    noise = noise / linalg.norm(noise) * linalg.norm(Xomega_ell) * 10 ** (-P / 20.)
    Xomega_ell_noisy = Xomega_ell + noise

    # noise energy, in the noiseless case 1e-14 is considered as 0
    # noise_level = np.max([1e-14, linalg.norm(noise)])
    # max_ini = 50  # maximum number of random initialisations

    G = np.eye(L)
    xhat_recon, min_error, c_opt = dirac_recon_time(G, Xomega_ell_noisy, K)[:3]
    # reconstruct Diracs' locations tk
    z = np.roots(c_opt)
    z = z / np.abs(z)
    tk_recon = np.real(1j / omega_step_sz * np.log(z))
    tk_recon[tk_recon >= 0.5 * tau] -= tau

    # location estimation error
    t_error = distance(tk_recon, tk)[0]

    # reconstruct amplitudes ak
    G_amp = np.exp(-1j * omega_ell.reshape((-1, 1)) * tk_recon.reshape((1, -1)))
    ak_recon = linalg.lstsq(G_amp, Xomega_ell_noisy)[0]
    if np.isclose(np.max(np.abs(ak.imag)), 0):
        ak_recon = ak_recon.real

    # plot reconstruction
    plt.close()
    fig = plt.figure(num=1, figsize=(5, 4), dpi=90)

    subplt_height = 0.2
    subplt_width = 0.87
    subplt_left_corner = 0.115
    # sub-figure 1
    ax1 = plt.axes([subplt_left_corner, 0.71, subplt_width, subplt_height])
    markerline311_1, stemlines311_1, baseline311_1 = ax1.stem(tk, ak, label='Original Diracs')
    plt.setp(stemlines311_1, linewidth=1.5, color=[0, 0.447, 0.741])
    plt.setp(markerline311_1, marker='^', linewidth=1.5, markersize=8,
             markerfacecolor=[0, 0.447, 0.741], mec=[0, 0.447, 0.741])
    plt.setp(baseline311_1, linewidth=0)

    markerline311_2, stemlines311_2, baseline311_2 = \
        plt.stem(tk_recon, ak_recon, label='Estimated Diracs')
    plt.setp(stemlines311_2, linewidth=1.5, color=[0.850, 0.325, 0.098])
    plt.setp(markerline311_2, marker='*', linewidth=1.5, markersize=10,
             markerfacecolor=[0.850, 0.325, 0.098], mec=[0.850, 0.325, 0.098])
    plt.setp(baseline311_2, linewidth=0)
    ax1.yaxis.set_tick_params(labelsize=8.5)
    plt.axhline(0, color='k')
    plt.xlim([-tau / 2, tau / 2])
    plt.ylim([1.18 * np.min(np.concatenate((ak, ak_recon, np.array(0)[np.newaxis]))),
              1.18 * np.max(np.concatenate((ak, ak_recon, np.array(0)[np.newaxis])))])
    plt.xlabel(r'$t$', fontsize=12)
    plt.ylabel(r'amplitudes', fontsize=12)
    ax1.xaxis.set_label_coords(0.5, -0.21)
    plt.legend(numpoints=1, loc=0, fontsize=9, framealpha=0.3,
               columnspacing=1.7, labelspacing=0.1)
    t_error_pow = np.int(np.floor(np.log10(t_error)))
    plt.title(r'$K={0}$, $L={1}$, '
              r'$\mbox{{SNR}}={2}$dB, '
              r'$t_{{\mbox{{\footnotesize error}}}}='
              r'{3:.2f}\times10^{other}$'.format(repr(K), repr(L), repr(P),
                                                 t_error / 10 ** t_error_pow,
                                                 other='{' + str(t_error_pow) + '}'),
              fontsize=12)
    # sub-figure 2
    omega_continuous = np.linspace(-np.pi * M, np.pi * M, num=np.max([10 * L, 10000]))
    tk_grid_conti, omega_grid_conti = np.meshgrid(tk, omega_continuous)
    Xomegas_conti = np.dot(np.exp(-1j * omega_grid_conti * tk_grid_conti), ak)

    G_conti_recon = np.exp(-1j * omega_continuous.reshape((-1, 1)) * tk_recon.reshape((1, -1)))
    Xomegas_conti_recon = np.dot(G_conti_recon, ak_recon)

    ax2 = plt.axes([subplt_left_corner, 0.358, subplt_width, subplt_height])
    line312_1 = ax2.plot(omega_ell, np.real(Xomega_ell_noisy), label='Measurements')
    plt.setp(line312_1, marker='.', linestyle='None', markersize=5, color=[0, 0.447, 0.741])

    line312_2 = plt.plot(omega_continuous, np.real(Xomegas_conti), label='Ground Truth')
    plt.setp(line312_2, linestyle='-', color=[0.850, 0.325, 0.098], linewidth=1)

    line312_3 = plt.plot(omega_continuous, np.real(Xomegas_conti_recon), label='Reconstruction')
    plt.setp(line312_3, linestyle='--', color=[0.466, 0.674, 0.188], linewidth=1.5)
    plt.ylim([1.1 * np.min(np.concatenate((np.real(Xomegas_conti), np.real(Xomega_ell_noisy)))),
              1.1 * np.max(np.concatenate((np.real(Xomegas_conti), np.real(Xomega_ell_noisy))))])
    ax2.yaxis.major.locator.set_params(nbins=7)
    plt.ylabel(r'$\Re\left\{X(\omega)\right\}$', fontsize=13)
    plt.legend(numpoints=1, loc=4, bbox_to_anchor=(1.013, 0.975), fontsize=9,
               handletextpad=.2, columnspacing=1.7, labelspacing=0.1, ncol=3)
    # sub-figure 3
    ax3 = plt.axes([subplt_left_corner, 0.10, subplt_width, subplt_height])
    line313_1 = ax3.plot(omega_ell, np.imag(Xomega_ell_noisy), label='Measurements')
    plt.setp(line313_1, marker='.', linestyle='None', markersize=5, color=[0, 0.447, 0.741])

    line313_2 = plt.plot(omega_continuous, np.imag(Xomegas_conti), label='Ground Truth')
    plt.setp(line313_2, linestyle='-', color=[0.850, 0.325, 0.098], linewidth=1)

    line313_3 = plt.plot(omega_continuous, np.imag(Xomegas_conti_recon), label='Reconstruction')
    plt.setp(line313_3, linestyle='--', color=[0.466, 0.674, 0.188], linewidth=1.5)
    plt.ylim([1.1 * np.min(np.concatenate((np.imag(Xomegas_conti), np.imag(Xomega_ell_noisy)))),
              1.1 * np.max(np.concatenate((np.imag(Xomegas_conti), np.imag(Xomega_ell_noisy))))])
    ax3.yaxis.major.locator.set_params(nbins=7)
    plt.ylabel(r'$\Im\left\{X(\omega)\right\}$', fontsize=12)
    plt.xlabel(r'$\omega$', fontsize=12)
    ax3.xaxis.set_label_coords(0.5, -0.21)

    if save_fig:
        if periodic_spectrum:
            file_name = (r'./result/TSP_eg2_K_{0}_L_{1}_M_{2}_noise_{3}dB_' +
                         interp_kernel + r'_periodic' +
                         r'.' + fig_format).format(repr(K), repr(L), repr(M), repr(P))
        else:
            file_name = (r'./result/TSP_eg2_K_{0}_L_{1}_M_{2}_noise_{3}dB_' +
                         interp_kernel + r'_aperiodic' +
                         r'.' + fig_format).format(repr(K), repr(L), repr(M), repr(P))
        plt.savefig(file_name, format=fig_format, dpi=300, transparent=True)
    plt.show()
