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

import bokeh.plotting as b_plt
from bokeh.io import vplot, hplot, output_file, show

from alg_tools_2d import gen_samples_edge_img, build_G_fri_curve, snr_normalised, \
    std_normalised, cadzow_iter_fri_curve, slra_fri_curve,\
    plt_recon_fri_curve, lsq_fri_curve, recon_fri_curve

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + \
                     ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True

if __name__ == '__main__':
    # various experiment settings
    save_fig = True
    fig_format = r'png'  # file type used to save the figure, e.g., pdf, png, etc.
    stop_cri = 'max_iter'  # stopping criteria: 1) mse; or 2) max_iter
    web_fig = False  # generate html file for the figures
    # dimension of the curve coefficients (2 * K0 + 1) (2 * L0 + 1)
    K0 = 1
    L0 = 1
    K = 2 * K0 + 1
    L = 2 * L0 + 1

    tau_x = 1  # period along x-axis
    tau_y = 1  # period along y-axis

    # curve coefficients
    stored_param = np.load('./data/coef.npz')
    coef = stored_param['coef']
    # coef = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    assert K == coef.shape[1] and L == coef.shape[0]

    # P = float('inf')  # noise level SNR (dB)
    P = 5
    M0 = 22  # number of sampling points along x-axis is 2 * M0 + 1 (at least 2 * K0)
    N0 = 22  # number of sampling points along y-axis is 2 * N0 + 1 (at least 2 * L0)
    samples_size = np.array([2 * N0 + 1, 2 * M0 + 1])
    # bandwidth of the ideal lowpass filter
    B_x = 25  #(2. * M0 + 1.) / tau_x#
    B_y = 25  #(2. * N0 + 1.) / tau_y#
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
    # samples_noiseless = gen_samples_edge_img(coef, samples_size, tau_x, tau_y)[0]
    samples_noiseless, fourier_lowpass = \
        gen_samples_edge_img(coef, samples_size, B_x, B_y, tau_x, tau_y)
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

    # least square reconstruction
    coef_recon_lsq = lsq_fri_curve(G, samples_noisy, K, L, B_x, B_y, tau_x, tau_y)
    std_lsq = std_normalised(coef_recon_lsq, coef)[0]
    snr_lsq = snr_normalised(coef_recon_lsq, coef)

    # cadzow iterative denoising
    K_cad = np.int(np.floor((B_x * tau_x - 1) / 4) * 2 + 1)
    L_cad = np.int(np.floor((B_y * tau_y - 1) / 4) * 2 + 1)
    coef_recon_cadzow = cadzow_iter_fri_curve(G, samples_noisy, K, L, K_cad, L_cad,
                                              B_x, B_y, tau_x, tau_y, max_iter=1000)
    std_cadzow = std_normalised(coef_recon_cadzow, coef)[0]
    snr_cadzow = snr_normalised(coef_recon_cadzow, coef)

    # structured low rank approximation (SLRA) by L. Condat
    K_alg = np.int(np.floor((B_x * tau_x - 1) / 4) * 2 + 1)
    L_alg = np.int(np.floor((B_y * tau_y - 1) / 4) * 2 + 1)
    # weight_choise: '1': the default one based on the repetition of entries in
    # the block Toeplitz matrix
    # weight_choise: '2': based on the repetition of entries in the block Toeplitz
    # matrix and the frequency re-scaling factor in hat_partial_I
    # weight_choise: '3': equal weights for all entries in the block Toeplitz matrix
    coef_recon_slra = slra_fri_curve(G, samples_noisy, K, L, K_alg, L_alg,
                                     B_x, B_y, tau_x, tau_y, max_iter=1000,
                                     weight_choice='1')
    std_slra = std_normalised(coef_recon_slra, coef)[0]
    snr_slra = snr_normalised(coef_recon_slra, coef)

    # the proposed approach
    max_ini = 20  # maximum number of random initialisations
    xhat_recon, min_error, coef_recon, ini = \
        recon_fri_curve(G, samples_noisy, K, L,
                        B_x, B_y, tau_x, tau_y, noise_level, max_ini, stop_cri)

    std_coef_error = std_normalised(coef_recon, coef)[0]
    snr_error = snr_normalised(coef_recon, coef)

    # print out results
    print('Least Square Minimisation')
    print('Standard deviation of the reconstructed ' +
          'curve coefficients error: {:.4f}'.format(std_lsq))
    print('SNR of the reconstructed ' +
          'curve coefficients: {:.4f}[dB]\n'.format(snr_lsq))

    print('Cadzow Iterative Method')
    print('Standard deviation of the reconstructed ' +
          'curve coefficients error: {:.4f}'.format(std_cadzow))
    print('SNR of the reconstructed ' +
          'curve coefficients: {:.4f}[dB]\n'.format(snr_cadzow))

    print('SLRA Method')
    print('Standard deviation of the reconstructed ' +
          'curve coefficients error: {:.4f}'.format(std_slra))
    print('SNR of the reconstructed ' +
          'curve coefficients: {:.4f}[dB]\n'.format(snr_slra))

    print('Proposed Approach')
    print('Standard deviation of the reconstructed ' +
          'curve coefficients error: {:.4f}'.format(std_coef_error))
    print('SNR of the reconstructed ' +
          'curve coefficients: {:.4f}[dB]\n'.format(snr_error))

    # plot results
    # spatial domain samples
    fig = plt.figure(num=0, figsize=(3, 3), dpi=90)
    plt.imshow(np.abs(samples_noisy), origin='upper', cmap='gray')
    plt.axis('off')
    if save_fig:
        file_name = (r'./result/TSP_eg3_K_{0}_L_{1}_' +
                     r'noise_{2}dB_samples.' + fig_format).format(repr(K), repr(L), repr(P))
        plt.savefig(file_name, format=fig_format, dpi=300, transparent=True)

    # Cadzow denoising result
    file_name = (r'./result/TSP_eg3_K_{0}_L_{1}_' +
                 r'noise_{2}dB_cadzow.' + fig_format).format(repr(K), repr(L), repr(P))
    curve_recon_cad = \
        plt_recon_fri_curve(coef_recon_cadzow, coef, tau_x, tau_y,
                            plt_size, save_fig, file_name, nargout=1,
                            file_format=fig_format)[0]

    # SLRA result
    file_name = (r'./result/TSP_eg3_K_{0}_L_{1}_' +
                 r'noise_{2}dB_slra.' + fig_format).format(repr(K), repr(L), repr(P))
    curve_recon_slra = \
        plt_recon_fri_curve(coef_recon_slra, coef, tau_x, tau_y,
                            plt_size, save_fig, file_name, nargout=1,
                            file_format=fig_format)[0]

    # proposed approach result
    file_name = ('./result/TSP_eg3_K_{0}_L_{1}_' +
                 'noise_{2}dB_proposed.' + fig_format).format(repr(K), repr(L), repr(P))
    curve_recon_proposed, idx_x, idx_y, subset_idx = \
        plt_recon_fri_curve(coef_recon, coef, tau_x, tau_y,
                            plt_size, save_fig, file_name, nargout=4,
                            file_format=fig_format)
    plt.show()

    if web_fig:
        output_file('./html/eg3.html')
        TOOLS = 'pan, wheel_zoom, reset'
        p_hdl1 = b_plt.figure(title=r'Noisy Samples (SNR = {:.1f}dB)'.format(P),
                              tools=TOOLS,
                              plot_width=320, plot_height=320,
                              x_range=(0, samples_size[1]),
                              y_range=(0, samples_size[0])
                              )
        p_hdl1.title.text_font_size['value'] = '12pt'
        p_hdl1.image(image=[samples_noisy], x=[0], y=[0],
                     dw=[samples_size[1]], dh=[samples_size[0]],
                     palette='Greys9')
        p_hdl1.axis.visible = None

        p_hdl2 = b_plt.figure(title=r'Cadzow''s Method',
                              tools=TOOLS,
                              plot_width=320, plot_height=320,
                              x_range=(0, plt_size[1]),
                              y_range=(0, plt_size[0])
                              )
        p_hdl2.title.text_font_size['value'] = '12pt'
        p_hdl2.image(image=[curve_recon_cad], x=[0], y=[0],
                     dw=[plt_size[1]], dh=[plt_size[0]],
                     palette='Greys9')
        p_hdl2.circle(x=idx_x[subset_idx], y=idx_y[subset_idx],
                      color='#D95319',
                      fill_color='#D95319',
                      line_width=1, size=1)
        p_hdl2.axis.visible = None

        p_hdl3 = b_plt.figure(title=r'SLRA Method',
                              tools=TOOLS,
                              plot_width=320, plot_height=320,
                              x_range=(0, plt_size[1]),
                              y_range=(0, plt_size[0])
                              )
        p_hdl3.title.text_font_size['value'] = '12pt'
        p_hdl3.image(image=[curve_recon_slra], x=[0], y=[0],
                     dw=[plt_size[1]], dh=[plt_size[0]],
                     palette='Greys9')
        p_hdl3.circle(x=idx_x[subset_idx], y=idx_y[subset_idx],
                      color='#D95319',
                      fill_color='#D95319',
                      line_width=1, size=1)
        p_hdl3.axis.visible = None

        p_hdl4 = b_plt.figure(title=r'Proposed',
                              tools=TOOLS,
                              plot_width=320, plot_height=320,
                              x_range=p_hdl2.x_range,
                              y_range=p_hdl2.y_range
                              )
        p_hdl4.title.text_font_size['value'] = '12pt'
        p_hdl4.image(image=[curve_recon_proposed], x=[0], y=[0],
                     dw=[plt_size[1]], dh=[plt_size[0]],
                     palette='Greys9')
        p_hdl4.circle(x=idx_x[subset_idx], y=idx_y[subset_idx],
                      color='#D95319',
                      fill_color='#D95319',
                      line_width=1, size=1)
        p_hdl4.axis.visible = None

        p_hdl = b_plt.gridplot([[p_hdl1, p_hdl2, p_hdl3, p_hdl4]],
                               toolbar_location='above')
        show(p_hdl)
