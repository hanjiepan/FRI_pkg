Towards Generalized FRI Sampling with an Application to Source Resolution in Radioastronomy
===========================================================================================

Authors
-------

Hanjie Pan<sup>1</sup>, Thierry Blu<sup>2</sup> and Martin Vetterli<sup>1</sup><br>
<sup>1</sup>Audiovisual Communications Laboratory ([LCAV](http://lcav.epfl.ch)) at [EPFL](http://www.epfl.ch).<br>
<sup>2</sup>Image and Video Processing Laboratory ([IVP](http://www.ee.cuhk.edu.hk/~tblu)) at [CUHK](http://www.cuhk.edu.hk).

<img src="http://lcav.epfl.ch/files/content/sites/lcav/files/images/Home/LCAV_anim_200.gif">

#### Contact

[Hanjie Pan](mailto:hanjie[dot]pan[at]epfl[dot]ch)<br>
EPFL-IC-LCAV<br>
BC 322 (Bâtiment BC)<br>
Station 14<br>
1015 Lausanne

Abstract
--------

Abstract—It is a classic problem to estimate continuous time sparse signals, like point sources in a direction-of-arrival problem, or pulses in a time-of-flight measurement. The earliest occurrence is the estimation of sinusoids in time series using Prony’s method. This is at the root of a substantial line of work on high resolution spectral estimation. The estimation of continuous-time sparse signals from discretetime samples is the goal of the sampling theory for finite rate of innovation (FRI) signals. Both spectral estimation and FRI sampling usually assume uniform sampling. 

But not all measurements are obtained uniformly, as exemplified by a concrete radioastronomy problem we set out to solve. Thus, we develop the theory and algorithm to reconstruct sparse signals, typically sum of sinusoids, from non-uniform samples. We achieve this by identifying a linear transformation that relates the unknown uniform samples of sinusoids to the given measurements. These uniform samples are known to satisfy the annihilation equations. A valid solution is then obtained by solving a constrained minimization such that the reconstructed signal is consistent with the given measurements and satisfies the annihilation constraint.

Thanks to this new approach, we unify a variety of FRI based methods. We demonstrate the versatility and robustness of the proposed approach with five FRI reconstruction problems, namely Dirac reconstructions with irregular time or Fourier domain samples, FRI curve reconstructions, Dirac reconstructions on the sphere and point source reconstructions in radioastronomy. The proposed algorithm improves substantially over state of the art methods and is able to reconstruct point sources accurately from irregularly sampled Fourier measurements under severe noise conditions.

Package Description
-------------------

This repository contains all the code to reproduce the results of the paper [*Towards Generalized FRI Sampling with an Application to Source Resolution in Radioastronomy*](http://lcav.epfl.ch). It contains a python implementation of the proposed algorithm.

A number of scripts were written to exemplify the application of the proposed *finite rate of innovation (FRI)* reconstruction algorithm on solving various FRI reconstruction problems, including

* Dirac reconstruction from *non-uniformly* sampled time domain samples (`fig6a_irreg_time_samp.py`, `fig6b_irreg_time_samp.py`)
* Dirac reconstruction from *non-uniformly* sampled Fourier domain samples (`fig7_irreg_fourier_samp.py`, `fig8a_irreg_fourier_samp.py`, `fig8b_irreg_fourier_samp.py`)
* FRI curve reconstruction (`fig9_fri_curve.py`, `fig10_fri_curve.py`)
* Point source reconstruction in radioastronomy (`fig2_radio_ast_eg1.py`, `fig11abc_radio_ast_eg2.py`, `fig11d_radio_ast_eg2.py`)

We are available for questions or requests relating to either the code or the theory behind it. 

Recreate the figures
--------------------------------------

Start an ipython/jupyter qt-console

    jupyter qtconsole
    
and then type in the following commands in an ipython shell.

    # Start inline plotting mode
    %matplotlib inline
    
    # Plot the radioastronomy example in the introduction
    %run fig2_radio_ast_eg1.py

    # Dirac reconstruction from non-uniformly sampled 
    # time domain measurements (noiseless and noisy cases)
    %run fig6a_irreg_time_samp.py
    %run fig6b_irreg_time_samp.py

    # Dirac reconstruction from non-uniformly sampled 
    # Fourier domain measurements
    # 1) the spectrum satisfies the periodic assumption
    %run fig7_irreg_fourier_samp.py
    # 2) the spectrum DOES NOT satisfy the periodic assumption 
    # (noiseless and noisy cases)
    %run fig8a_irreg_fourier_samp.py
    %run fig8b_irreg_fourier_samp.py

    # FRI curve reconstruction
    # 1) Monte Carlo simulation for different noise levels
    %run fig9_fri_curve.py
    # 2) A visual example for FRI curve reconstruction
    %run fig10_fri_curve.py

    # Point source reconstruction for radioastronomy
    %run fig11abc_radio_ast_eg2.py
    %run fig11d_radio_ast_eg2.py

Data used in the paper
----------------------

The randomly generated signal parameters are saved under the folder `data/`.

    # Simulation with non-uniform sampled time domain measurements
    data/Dirac_Data_20-12_02_22.npz

    # Simulation with non-uniform sampled Fourier measurements
    # periodic spectrum
    data/freq_Dirac_Data_p20-12_11_48.npz
    # aperiodic spectrum
    data/freq_Dirac_Data_ap20-12_11_55.npz

    # Experiment with FRI curve
    data/coef.npz

    # Experiments for radioastronomy
    data/Dirac_Data_8-2_19_47.npz
    data/Dirac_Data_4-2_02_28.npz

Saved simulation results
-------------

The Monte Carlo simulation results are saved in folder `result/`

* `avg_perf_K_3_L_3_all.npz`: FRI curve reconstruction with different levels of noise
* `radio_ast_batch_res1000.npz`: Point source reconstruction (3 sources)
* `radio_ast_eg3_batch_res1000.npz`: Point source reconstruction (5 sources)

Overview of results
-------------------

We apply the proposed algorithmic framework for several FRI reconstruction problems.

### Dirac reconstruction from time domain samples (Fig. 6)

We demonstrate how Diracs are reconstructed from non-uniformly sampled time 
domain measurements.

* the noiseless case:  
    <img src="./result/TSP_eg1_K_5_L_11_noise_infdB20-12_02_22.png" height="185">

* the noisy case:  
    <img src="./result/TSP_eg1_K_5_L_81_noise_5dB20-12_02_22.png" height="185">

### Dirac reconstruction from Fourier domain samples (Fig. 7 and Fig. 8)

We demonstrate how Diracs are reconstructed from non-uniformly sampled time domain measurements. 
We consider both cases where the spectrum is periodic and aperiodic. 
The difference is whether we have an exact mapping from a set of unknown uniform samples of sinusoids to the given non-uniformly sampled measurements.

* In the case with periodic spectrum, the linear mapping is exact.  
    <img src="./result/TSP_eg2_K_5_L_21_M_21_noise_infdB_dirichlet_periodic.png" height="320">

* While in the case with aperiodic spectrum, we use interpolation to approximate the linear mapping.  
    <img src="./result/TSP_eg2_K_5_L_42_M_21_noise_infdB_triangular_aperiodic.png" height="320">
    <img src="./result/TSP_eg2_K_5_L_105_M_21_noise_5dB_triangular_aperiodic.png" height="320">

### FRI curve reconstruction (Fig. 9 and Fig. 10)

We demonstrate the flexibilities of the proposed algorithmic framework in 
choosing a formulation that is more suitable for a given FRi reconstruction problem.  
<img src="./result/avg_perf_plot_K_3_L_3.png" height="250">

Visiual comparison for (SNR=5dB)  
<table>
  <tr>
    <td align="center"><img src="./result/TSP_eg3_K_3_L_3_noise_5dB_samples.png" height="200"></td>
    <td align="center"><img src="./result/TSP_eg3_K_3_L_3_noise_5dB_cadzow.png" height="200"></td>
  </tr>
  <tr>
    <td align="center">(a) Noisy Samples</td>
    <td align="center">(b) Cadzow's Method</td>
  </tr>
  <tr>
  <td align="center"><img src="./result/TSP_eg3_K_3_L_3_noise_5dB_slra.png" height="200"></td>
    <td align="center"><img src="./result/TSP_eg3_K_3_L_3_noise_5dB_proposed.png" height="200"></td>
  </tr>
  <tr>
    <td align="center">(c) Structured Low-rank Approximation</td>
    <td align="center">(d) Proposed Approach</td>
  </tr>
</table>
 
  

### Point source reconstruction in radioastronomy (Fig. 2 and Fig. 11)

We demonstrate one application of the proposed FRI-based reconstruction 
algorithm on reconstructing point sources in radioastronomy.

* Resolve closely located sources (SNR = 5dB)  
    <table>
      <colgroup>
          <col width="250">
          <col width="250">
      </colgroup>
      <tr>
        <td align="center"><img src="./result/TSP_intro_K_3_L_8000_noise_5dB_dirty_img_515by515.png" height="200"></td>
        <td align="center"><img src="./result/TSP_intro_K_3_L_8000_noise_5dB_aggregated.png" height="200"></td>
      </tr>
      <tr>
        <td align="center">(a) Inverser Fourier Transform of the Raw Measurements</td>
        <td align="center">(b) *Statistics* of Retrieved Sources using FRI</td>
      </tr>
      <tr>
        <td align="center"><img src="./result/TSP_intro_K_3_L_8000_noise_5dB_dirty_img_515by515_zoom.png" height="200"></td>
        <td align="center"><img src="./result/TSP_intro_K_3_L_8000_noise_5dB_aggregated_zoom.png" height="200"></td>
      </tr>
      <tr>
        <td align="center">(c) Zoom-in of (a)</td>
        <td align="center">(d) Zoom-in of (b)</td>
      </tr>
    </table>

* Comparison with classic approaches
    <table>
      <colgroup>
          <col width="250">
          <col width="250">
          <col width="250">
      </colgroup>
      <tr>
        <td align="center"><img src="./result/TSP_eg4_K_5_L_8500_noise_5dB_spectrum_noisy.png" height="200"></td>
        <td align="center"><img src="./result/TSP_eg4_K_5_L_8500_noise_5dB_dirty_img_515by515.png" height="200"></td>
        <td align="center"><img src="./result/TSP_eg4_K_5_L_8500_noise_5dB_ell1_recon_515by515.png" height="200"></td>
      </tr>
    </table>
    <table>
        <colgroup>
          <col width="500">
          <col width="250">
      </colgroup>
      <tr>
        <td align="center">(a) The Given Measurements</td>
        <td align="center">(b) &#8467;<sub>1</sub> Minimization Result</td>
      </tr>
      <tr>
        <td align="center"><img src="./result/TSP_eg4_K_5_L_8500_noise_5dB_locations.png" height="163"></td>
        <td align="center"><img src="./result/TSP_intro_K_5_L_8500_noise_5dB_aggregated.png" height="200"></td>
      </tr>
      <tr>
        <td align="center">(c) *Continuous* Domain Reconstruction (FRI)</td>
        <td align="center">(d) *Statistics* of the Retrieved Sources Locations (FRI)</td>
      </tr>
    </table>

Dependencies
------------

* A working distribution of [Python 3](https://www.python.org/downloads/).
* [Numpy](http://www.numpy.org/), [Scipy](http://www.scipy.org/)
* We use the distribution [anaconda](https://store.continuum.io/cshop/anaconda/) to simplify the setup of the environment.
* We use the [MKL](https://store.continuum.io/cshop/mkl-optimizations/) extension of Anaconda to speed things up. There is a [free license](https://store.continuum.io/cshop/academicanaconda) for academics.
* We use joblib for parallel computations.
* [matplotlib](http://matplotlib.org) for plotting the results.

List of standard packages needed

    numpy, scipy, matplotlib, mkl, joblib


Systems Tested
--------------

### OS X

| Machine | MacBook Pro Retina 15-inch, Mid 2014 |
|---------|----------------------------------------|
| System  | macOS Sierra 10.12                     |
| CPU     | Intel Core i7                          |
| RAM     | 16 GB                                  |

    System Info:
    ------------
    Darwin Kernel Version 16.0.0: Mon Aug 29 17:56:20 PDT 2016; root:xnu-3789.1.32~3/RELEASE_X86_64 x86_64

    Python Info:
    ------------
    Python 3.5.2 :: Anaconda custom (x86_64)

    Python Packages Info (conda)
    ----------------------------
    # packages in environment at /Users/pan/anaconda:
    accelerate                2.3.0               np111py35_3  
    accelerate_cudalib        2.0                           0  
    anaconda                  custom                   py35_0  
    anaconda-client           1.4.0                    py35_0  
    anaconda-navigator        1.3.1                    py35_0  
    joblib                    0.9.4                    py35_0  
    jupyter                   1.0.0                    py35_3  
    jupyter_client            4.3.0                    py35_0  
    jupyter_console           4.1.1                    py35_0  
    jupyter_core              4.1.0                    py35_0  
    matplotlib                1.5.3               np111py35_0  
    mkl                       11.3.3                        0  
    mkl-service               1.1.2                    py35_2  
    nb_anacondacloud          1.1.0                    py35_0  
    numpy                     1.11.1                   py35_0  
    pyqt                      5.6.0                    py35_0  
    qt                        5.6.0                         0  
    qtawesome                 0.3.3                    py35_0  
    qtconsole                 4.2.1                    py35_1  
    qtpy                      1.1.2                    py35_0  
    scipy                     0.18.1              np111py35_0 

License
-------

Copyright (c) 2016, Hanjie Pan, Thierry Blu and Martin Vetterli<br>
The source code is released under the [MIT](https://opensource.org/licenses/MIT) license.
