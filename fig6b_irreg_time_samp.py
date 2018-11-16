from __future__ import division
import datetime
import os
import numpy as np
from scipy import linalg
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
else:
    matplotlib.use('Qt5Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
# import bokeh.plotting as b_plt
# from bokeh.io import vplot, hplot, output_file, show
# from bokeh.models.tools import WheelZoomTool

from alg_tools_1d import dirac_recon_time, periodicSinc, distance

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + \
                     ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True

if __name__ == '__main__':
    # various experiment settings
    save_fig = True  # save figure or not
    fig_format = r'png'  # file type used to save the figure, e.g., pdf, png, etc.
    stop_cri = 'max_iter'  # stopping criteria: 1) mse; or 2) max_iter
    web_fig = False  # generate html file for the figures

    K = 5  # number of Diracs
    M = K * 8  # number of Fourier samples (at least K)
    tau = 1  # period of the Dirac stream

    # number of time domain samples
    L = (2 * M + 1)
    Tmax = tau / L  # the average sampling step size (had we used a uniform sampling setup)

    # generate the random sampling time instances
    t_samp = np.arange(0, L, dtype=float) * Tmax
    t_samp += np.sign(np.random.randn(L)) * np.random.rand(L) * Tmax / 2.
    # round t_samp to [0, tau)
    t_samp -= np.floor(t_samp / tau) * tau

    # generate parameters for the periodic stream of Diracs
    B = (2. * M + 1.) / tau  # bandwidth of the sampling filter

    '''
    # generate random values for Dirac amplitudes and locations
    # amplitudes of the Diracs
    ak = np.sign(np.random.randn(K)) * (1 + (np.random.rand(K) - 0.5) / 1.)
    # locations of the Diracs
    if K == 1:
        tk = np.random.rand()
    else:
        a = 4. / L
        uk = np.random.exponential(scale=1. / K, size=(K - 1, 1))
        tk = np.cumsum(a + (1. - K * a) * (1 - 0.1 * np.random.rand()) / uk.sum() * uk)
        tk = np.sort(np.hstack((np.random.rand() * tk[0] / 2., tk)) + (1 - tk[-1]) / 2.) * tau

        # save Dirac parameter
        time_stamp = datetime.datetime.now().strftime("%-d-%-m_%H_%M")
        file_name = './data/Dirac_Data_' + time_stamp + '.npz'
        np.savez(file_name, tk=tk, ak=ak, K=K, time_stamp=time_stamp)
    '''

    # load saved data
    time_stamp = '20-12_02_22'
    stored_param = np.load('./data/Dirac_Data_' + time_stamp + '.npz')
    tk = stored_param['tk']
    ak = stored_param['ak']

    print('time stamp: ' + time_stamp +
          '\n=======================================\n')

    # compute the noiseless Fourier series coefficients
    tk_grid, m_grid_gt = np.meshgrid(tk, np.arange(-np.floor(B * tau / 2.), 1 + np.floor(B * tau / 2.)))
    x_hat_noiseless = 1. / tau * np.dot(np.exp(-2j * np.pi / tau * m_grid_gt * tk_grid), ak)

    m_grid, t_samp_grid = np.meshgrid(np.arange(-np.floor(B * tau / 2.), 1 + np.floor(B * tau / 2.)), t_samp)

    # build the linear transformation matrix that links x_hat with the samples
    G = 1. / B * np.exp(2j * np.pi / tau * m_grid * t_samp_grid)
    y_ell_noiseless = np.real(np.dot(G, x_hat_noiseless))

    # add noise
    P = 5
    noise = np.random.randn(L)
    noise = noise / linalg.norm(noise) * linalg.norm(y_ell_noiseless) * 10 ** (-P / 20.)
    y_ell = y_ell_noiseless + noise

    # noise energy, in the noiseless case 1e-10 is considered as 0
    noise_level = np.max([1e-10, linalg.norm(noise)])
    max_ini = 100  # maximum number of random initialisations

    # FRI reconstruction
    xhat_recon, min_error, c_opt, ini = dirac_recon_time(G, y_ell, K, noise_level, max_ini, stop_cri)

    print(r'Noise level: {0:.2e}'.format(noise_level))
    print(r'Minimum approximation error |a - Gb|_2: {0:.2e}'.format(min_error))

    # reconstruct Diracs' locations tk
    z = np.roots(c_opt)
    z = z / np.abs(z)
    tk_recon = np.real(tau * 1j / (2 * np.pi) * np.log(z))
    tk_recon = np.sort(tk_recon - np.floor(tk_recon / tau) * tau)

    # reconstruct amplitudes ak
    Phi_recon = periodicSinc(np.pi * B * (np.reshape(t_samp, (-1, 1), order='F') -
                                          np.reshape(tk_recon, (1, -1), order='F')),
                             B * tau)
    ak_recon = np.real(linalg.lstsq(Phi_recon, y_ell)[0])
    # location estimation error
    t_error = distance(tk_recon, tk)[0]
    # plot reconstruction
    plt.close()
    fig = plt.figure(num=1, figsize=(5.5, 2.5), dpi=90)
    # sub-figure 1
    ax1 = plt.axes([0.125, 0.59, 0.85, 0.31])
    markerline211_1, stemlines211_1, baseline211_1 = \
        ax1.stem(tk, ak, label='Original Diracs')
    plt.setp(stemlines211_1, linewidth=1.5, color=[0, 0.447, 0.741])
    plt.setp(markerline211_1, marker='^', linewidth=1.5, markersize=8,
             markerfacecolor=[0, 0.447, 0.741], mec=[0, 0.447, 0.741])
    plt.setp(baseline211_1, linewidth=0)

    markerline211_2, stemlines211_2, baseline211_2 = \
        plt.stem(tk_recon, ak_recon, label='Estimated Diracs')
    plt.setp(stemlines211_2, linewidth=1.5, color=[0.850, 0.325, 0.098])
    plt.setp(markerline211_2, marker='*', linewidth=1.5, markersize=10,
             markerfacecolor=[0.850, 0.325, 0.098], mec=[0.850, 0.325, 0.098])
    plt.setp(baseline211_2, linewidth=0)

    plt.axhline(0, color='k')
    plt.xlim([0, tau])
    plt.ylim([1.17 * np.min(np.concatenate((ak, ak_recon, np.array(0)[np.newaxis]))),
              1.17 * np.max(np.concatenate((ak, ak_recon, np.array(0)[np.newaxis])))])
    # plt.xlabel(r'$t$', fontsize=12)
    plt.ylabel('amplitudes', fontsize=12)
    ax1.yaxis.set_label_coords(-0.095, 0.5)
    plt.legend(numpoints=1, loc=0, fontsize=9, framealpha=0.3,
               handletextpad=.2, columnspacing=0.6, labelspacing=0.05, ncol=2)
    t_error_pow = np.int(np.floor(np.log10(t_error)))
    if np.isinf(P):
        plt.title(r'$K={0}$, $L={1}$, '
                  r'$\mbox{{SNR}}=\mbox{{inf }}$dB, '
                  r'$t_{{\mbox{{\footnotesize err}}}}={2:.2f}\times10^{other}$'.format(repr(K), repr(L),
                                                                                         t_error / 10 ** t_error_pow,
                                                                                         other='{' + str(
                                                                                             t_error_pow) + '}'),
                  fontsize=12)
    else:
        plt.title(r'$K={0}$, $L={1}$, '
                  r'$\mbox{{SNR}}={2}$dB, '
                  r'$t_{{\mbox{{\footnotesize err}}}}={3:.2f}\times10^{other}$'.format(repr(K), repr(L), repr(P),
                                                                                         t_error / 10 ** t_error_pow,
                                                                                         other='{' + str(
                                                                                             t_error_pow) + '}'),
                  fontsize=12)

    # sub-figure 2
    t_plt = np.linspace(0, tau, num=np.max([10 * L, 1000]))
    m_plt_grid, t_plt_grid = np.meshgrid(np.arange(-np.floor(B * tau / 2.),
                                                   1 + np.floor(B * tau / 2.)),
                                         t_plt)
    G_plt = 1. / B * np.exp(2j * np.pi / tau * m_plt_grid * t_plt_grid)
    y_plt = np.real(np.dot(G_plt, x_hat_noiseless))  # for plotting purposes only

    ax2 = plt.axes([0.125, 0.18, 0.85, 0.31])
    line212_1 = ax2.plot(t_plt, y_plt, label='Ground Truth')
    plt.setp(line212_1, linestyle='-', color=[0, 0.447, 0.741], linewidth=1)

    line212_2 = ax2.plot(t_samp, y_ell, label='Samples')
    plt.setp(line212_2, marker='.', linestyle='None', markersize=5, color=[0.850, 0.325, 0.098])
    plt.ylim([1.05 * np.min(np.concatenate((y_plt, y_ell))),
              1.05 * np.max(np.concatenate((y_plt, y_ell)))])
    plt.ylabel(r'$x(t) * \mathrm{{sinc}}(B t)$', fontsize=12)
    plt.xlabel(r'$t$', fontsize=12)
    ax2.xaxis.set_label_coords(0.5, -0.21)
    ax2.yaxis.set_label_coords(-0.095, 0.5)
    plt.legend(numpoints=1, loc=0, fontsize=9, framealpha=0.3,
               handletextpad=.2, columnspacing=0.6, labelspacing=0.05, ncol=2)

    if save_fig:
        file_name = (r'./result/TSP_eg1_K_{0}_L_{1}_noise_{2}dB' +
                     time_stamp + r'.' + fig_format).format(repr(K), repr(L), repr(P))
        plt.savefig(file_name, format=fig_format, dpi=300, transparent=True)

    plt.show()

    # for web rendering
    # if web_fig:
    #     output_file('./html/eg1.html')
    #     TOOLS = 'pan,box_zoom,box_select,reset'
    #     p_hdl1 = b_plt.figure(title='K={0}, L={1}, SNR={2:.1f}dB, error={3:.2e}'.format(repr(K), repr(L), P, t_error),
    #                           tools=TOOLS,
    #                           x_axis_label='time', y_axis_label='amplitudes',
    #                           plot_width=550, plot_height=220,
    #                           x_range=(0, tau),
    #                           y_range=(1.17 * np.min(np.concatenate((ak, ak_recon,
    #                                                                  np.array(0)[np.newaxis]))),
    #                                    1.17 * np.max(np.concatenate((ak, ak_recon,
    #                                                                  np.array(0)[np.newaxis]))))
    #                           )
    #     p_hdl1.title.text_font_size = '12pt'
    #     p_hdl1.add_tools(WheelZoomTool(dimensions=["width"]))
    #     p_hdl1.triangle(x=tk, y=ak,
    #                     color='#0072BD',
    #                     fill_color='#0072BD',
    #                     line_width=1.5, size=8,
    #                     legend='Original Diracs')
    #     p_hdl1.multi_line(xs=np.vstack((tk, tk)).T.tolist(),
    #                       ys=np.vstack((np.zeros(ak.shape), ak)).T.tolist(),
    #                       color='#0072BD',
    #                       line_width=1.5,
    #                       line_color='#0072BD')
    #     p_hdl1.diamond(x=tk_recon, y=ak_recon,
    #                    color='#D95319',
    #                    line_width=1.5, size=10,
    #                    legend='Estimated Diracs')
    #     p_hdl1.multi_line(xs=np.vstack((tk_recon, tk_recon)).T.tolist(),
    #                       ys=np.vstack((np.zeros(ak_recon.shape), ak_recon)).T.tolist(),
    #                       color='#D95319',
    #                       line_width=1.5,
    #                       line_color='#D95319')
    #     p_hdl1.legend.location = 'bottom_right'
    #     p_hdl1.legend.border_line_alpha = 0.6
    #     p_hdl1.xaxis.axis_label_text_font_size = "11pt"
    #     p_hdl1.yaxis.axis_label_text_font_size = "11pt"
    #     p_hdl1.legend.legend_spacing = 1
    #     p_hdl1.legend.legend_padding = 5
    #     p_hdl1.legend.label_text_font_size = "9pt"
    #
    #     # subplot 2
    #     TOOLS2 = 'pan,box_zoom,wheel_zoom,box_select,reset'
    #     p_hdl2 = b_plt.figure(tools=TOOLS2, x_axis_label='time', y_axis_label='lowpssed signal',
    #                           plot_width=550, plot_height=220,
    #                           x_range=p_hdl1.x_range,
    #                           y_range=(1.05 * np.min(np.concatenate((y_plt, y_ell))),
    #                                    1.05 * np.max(np.concatenate((y_plt, y_ell))))
    #                           )
    #
    #     p_hdl2.line(x=t_plt, y=y_plt,
    #                 color='#0072BD',
    #                 line_color='#0072BD',
    #                 line_width=1.5,
    #                 legend='Ground Truth')
    #     p_hdl2.circle(x=t_samp, y=y_ell,
    #                   color='#D95319',
    #                   fill_color='#D95319',
    #                   line_width=1.5, size=2,
    #                   legend='Samples')
    #
    #     p_hdl2.xaxis.axis_label_text_font_size = "11pt"
    #     p_hdl2.yaxis.axis_label_text_font_size = "11pt"
    #     p_hdl2.legend.location = 'bottom_right'
    #     p_hdl2.legend.border_line_alpha = 0.6
    #     p_hdl2.legend.legend_spacing = 1
    #     p_hdl2.legend.legend_padding = 5
    #     p_hdl2.legend.label_text_font_size = "9pt"
    #
    #     p_hdl = b_plt.gridplot([[p_hdl1], [p_hdl2]], toolbar_location='above')
    #     show(p_hdl)
