from __future__ import division
import os
import numpy as np
import matplotlib

if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
else:
    matplotlib.use('Qt5Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from alg_tools_2d import gen_samples_edge_img, build_G_fri_curve, run_algs

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + \
                     ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True

if __name__ == '__main__':
    # various experiment settings
    max_noise_realisation = 500  # total number of different noise realisations
    snr_seq = [0, 2, 3, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    fig_format = r'png'  # file type used to save the figure, e.g., pdf, png, etc.
    # dimension of the curve coefficients (2 * K0 + 1) (2 * L0 + 1)
    K0 = 1
    L0 = 1
    K = 2 * K0 + 1
    L = 2 * L0 + 1

    tau_x = 1  # period along x-axis
    tau_y = 1  # period along y-axis

    # curve coefficients
    stored_param = np.load(r'./data/coef.npz')
    coef = stored_param['coef']
    assert K == coef.shape[1] and L == coef.shape[0]

    M0 = 22  # number of sampling points along x-axis is 2 * M0 + 1 (at least 2 * K0)
    N0 = 22  # number of sampling points along y-axis is 2 * N0 + 1 (at least 2 * L0)
    samples_size = np.array([2 * N0 + 1, 2 * M0 + 1])
    # bandwidth of the ideal lowpass filter
    B_x = 25
    B_y = 25

    # sampling step size
    T1 = tau_x / samples_size[1]  # along x-axis
    T2 = tau_y / samples_size[0]  # along y-axis

    # checking the settings
    assert (B_x * tau_x) % 2 == 1 and (B_y * tau_y) % 2 == 1
    assert B_x * T1 <= 1 and B_y * T2 <= 1
    assert (B_x * tau_x - K + 1) * (B_y * tau_y - L + 1) >= K * L

    # sampling locations
    x_samp = np.linspace(0, tau_x, num=samples_size[1], endpoint=False)
    y_samp = np.linspace(0, tau_y, num=samples_size[0], endpoint=False)
    # linear mapping between the spatial domain samples and the FRI sequence
    G = build_G_fri_curve(x_samp, y_samp, B_x, B_y, tau_x, tau_y)

    plt_size = np.array([1e3, 1e3])  # size for the plot of the reconstructed FRI curve

    # generate ideal samples
    samples_noiseless, fourier_lowpass = \
        gen_samples_edge_img(coef, samples_size, B_x, B_y, tau_x, tau_y)

    K_cad = np.int(np.floor((B_x * tau_x - 1) / 4.) * 2 + 1)
    L_cad = np.int(np.floor((B_y * tau_y - 1) / 4.) * 2 + 1)

    file_name_summary = r'./result/avg_perf_K_{0}_L_{1}_all.npz'.format(repr(K), repr(L))
    if os.path.isfile(file_name_summary):
        result_all = np.load(file_name_summary)
        avg_perf_summary = result_all['avg_perf_summary']
    else:
        avg_perf_summary = np.zeros((len(snr_seq), 9))
        counter = 0
        for P in snr_seq:
            batch_res = Parallel(n_jobs=-1)(delayed(run_algs)(coef, G, samples_noiseless,
                                                              P, K, L, K_cad, L_cad,
                                                              B_x, B_y, tau_x, tau_y,
                                                              max_iter_cadzow=1000,
                                                              max_iter_srla=1000,
                                                              max_ini=50)
                                            for realisation in range(max_noise_realisation))

            batch_res = np.asarray(batch_res)

            # compute the average performance
            avg_perf = np.mean(batch_res, axis=0)
            print('Noise level (SNR): {0}dB'.format(repr(P)))
            print('==========================================')
            print('Average Performance (Total Least Square)')
            print('std coef_error: {0}'.format(repr(avg_perf[0])))
            print('SNR(coef_recon, coef_ref): {0}[dB]\n'.format(repr(avg_perf[1])))

            print('Average Performance (Cadzow)')
            print('std coef_error: {0}'.format(repr(avg_perf[2])))
            print('SNR(coef_recon, coef_ref): {0}[dB]\n'.format(repr(avg_perf[3])))

            print('Average Performance (SLRA)')
            print('std coef_error: {0}'.format(repr(avg_perf[4])))
            print('SNR(coef_recon, coef_ref): {0}[dB]\n'.format(repr(avg_perf[5])))

            print('Average Performance (Proposed)')
            print('std coef_error: {0}'.format(repr(avg_perf[6])))
            print('SNR(coef_recon, coef_ref): {0}[dB]\n'.format(repr(avg_perf[7])))
            # save the result
            file_name = r'./result/avg_perf_K_{0}_L_{1}_snr_{2}dB.npz'.format(repr(K), repr(L), repr(P))
            np.savez(file_name, batch_res=batch_res, avg_perf=avg_perf, P=P)
            # store it in the matrix
            avg_perf_summary[counter, :-1] = avg_perf
            avg_perf_summary[counter, -1] = P
            counter += 1
            np.savez(file_name_summary, avg_perf_summary=avg_perf_summary)

    # plot results
    fig = plt.figure(figsize=(5.5, 3.3), dpi=90)
    plt.semilogy(avg_perf_summary[:, -1], avg_perf_summary[:, 0], label='Total Least Square',
                 zorder=10, linestyle=':', linewidth=2, color=[0.494, 0.184, 0.556],
                 markerfacecolor=[0.494, 0.184, 0.556], mec=[0.494, 0.184, 0.556])
    plt.semilogy(avg_perf_summary[:, -1], avg_perf_summary[:, 2], label=r"Cadzow's Method",
                 zorder=10, linestyle='--', linewidth=1, color=[0.850, 0.325, 0.098],
                 markerfacecolor=[0.850, 0.325, 0.098], mec=[0.850, 0.325, 0.098], alpha=0.75)
    plt.semilogy(avg_perf_summary[:, -1], avg_perf_summary[:, 4],
                 label='Structured Low-rank \n Approximation',
                 zorder=10, linestyle='-', linewidth=1, color=[0, 0.447, 0.741],
                 markerfacecolor=[0, 0.447, 0.741], mec=[0, 0.447, 0.741], alpha=0.5)
    plt.semilogy(avg_perf_summary[:, -1], avg_perf_summary[:, 6], label='Proposed',
                 zorder=10, linestyle='-.', linewidth=2, color=[0.466, 0.674, 0.188],
                 markerfacecolor=[0.466, 0.674, 0.188], mec=[0.466, 0.674, 0.188])

    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(b=True, zorder=0, which='major', linestyle='-', color=[0.7, 0.7, 0.7])
    plt.grid(b=True, zorder=0, which='minor', linestyle=':', color=[0.7, 0.7, 0.7])
    legend = plt.legend(numpoints=1, loc=0, fontsize=9,
                        columnspacing=1.7, labelspacing=0.5)
    plt.locator_params(axis='x', tight=True, nbins=11)
    plt.xlabel(r'noise level (SNR in [dB])', fontsize=11)
    plt.ylabel(r'$\mathrm{std}(\gamma{\rm\bf c}^\prime-{\rm\bf c})$', fontsize=12)
    plt.setp(legend.get_texts(), va='baseline')

    file_name = ('./result/avg_perf_plot_K_{0}_L_{1}.' + fig_format).format(repr(K), repr(L))
    plt.savefig(file_name, format=fig_format, dpi=300, transparent=True)
    plt.show()
