# Copyright 2018 Constantinos Papayiannis
#
# This file is part of Reverberation Learning Toolbox for Python.
#
# Reverberation Learning Toolbox for Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Reverberation Learning Toolbox for Python is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Reverberation Learning Toolbox for Python.  If not, see <http://www.gnu.org/licenses/>.

"""
Acoustic environment model object, based on the work in
C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation of Room Classifiers using Generative Adversarial Networks," arXiv preprint arXiv:1901.03257, 2018.

This file was original distributed in the repository at:
{repo}

If you use this code in your work, then cite:
C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation of Room Classifiers using Generative Adversarial Networks," arXiv preprint arXiv:1901.03257, 2018.

"""

from pickle import load, dump

import numpy as np
import os.path
from audiolazy.lazy_lpc import lpc
from scipy.optimize import brute, fmin, minimize
from scipy.signal import lfilter
from scipy.stats import norm
from sklearn.linear_model import ElasticNet

from third_party import prony
from utils_base import getfname, matmax, row_vector, column_vector, matmin, float2str, get_git_hash, \
    run_command
from utils_reverb import get_t60_decaymodel, get_edc, get_drr_linscale, air_up_to_db
from utils_spaudio import gm_frac_delayed_copies, enframe, overlapadd, get_array_energy, plotnorm


class AcEnvGenModelBase(object):
    """

    Acoustic environment model object, based on the work in
    C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation of Room Classifiers using Generative Adversarial Networks," arXiv preprint arXiv:1901.03257, 2018.

    It models the acoustic environment using a sparse model for the reflections in the early part
    and complements this model for a stochastic model for the tail. The tail model represents the
    decaying tail, which is filtered by an IIR filter to model room models and material
    absorptions.

    Attributes:
        acoustic_env: An instance of AcEnv. This contains the base acoustic environment to be
                            modeled
        air_reconstructed_from_model: The reconstructed AIR from the modeling result
        cache_dir: The location where the cached data is stored
        channel_to_model: The channel of the AIR to model
        defined_transition_time: The transition time between the early and late part
        direct_sound_neighborhood: The span of the direct sound region in seconds
        early_only: Flag to model only the early part
        early_opt_delayresolution: The resolution (in fractional samples) of the early part modeling
        early_opt_max_ref: The maximum reflections modeled per optimization step in the early part
        early_opt_spars_prom_factor: The sparsity promoting factor in the early part
        edc_compensation_enable: Enable compensation for the decay of the sound level in the modeling
        edc_scale: EDC scale compensator samples
        edc_scale_lim: The lower limit of the EDC scaler
        given_excitation: The provided excitation samples
        lasso_b: The predictors of the LASSO estimation for the early part initialization
        lasso_esigma_lambda: The proportion of the MSE standard deviation used to evaluate the
                            LASSO lambda for the regularization adjustment
        late_num_ar_coeff: The number of AR coefficients in the IIR filter for the tail
        late_num_ma_coeff: The number of MA coefficients in the IIR filter for the tail
        late_opt_delayresolution: The resolution (in fractional samples) of the late part modeling
        late_proc_framelength:The framelength for the late part modeling. Depends on available
                            memory
        late_tail_gaussian_std: The standard deviation of the White Gaussian Noise for the
                            stochastic tail model
        model_late_part: Flag for the modellign of the late part
        name: The name of this model
        number_of_parameters: Number of parameters used for this model
        original: The samples of the original AIR
        paramset_directsound_amp: The direct sound level
        paramset_directsound_soa: The ToA of the direct sound
        paramset_drr_linscale: The DRR in linear scale looking at all reflections
        paramset_drr_linscale_early: The DRR looking only at the early reflections
        paramset_drr_linscale_late: The DRR looking at only late reflections
        paramset_excitation_params: The excitation samples
        paramset_reflections_amp: The reflection scales
        paramset_reflections_fval_reduction: The reduction in the error offered by each reflection
        paramset_reflections_soa: The ToA of reflections
        paramset_reverberation_time: The T60 of the environment
        paramset_tail_modes_ar_coeff: The denominator coefficients of the IIR filter
        paramset_tail_modes_ma_coeff: The numerator coefficients of the IIR filter
        plot_exci: Flag for excitation plotting
        quiet: Flag for quiet processing
        sampling_freq: The samplign frequency
        single_side_dir_path_winlen: The single sided span of the direct sound
        soa_decimal_precision: The decimal precision of the ToA when expressed in fractional samples
        tail_visible_energy_ratio: The span of the AIR to model in terms of total energy
        transition_sample: The last sample index of the early part
        use_caches: Flag for cache usage
        use_delta_as_excitation: indicate the use of Dirac delta for the excitation (no channel)
        use_matlab: Flag for MATLAB engine usage


    """

    def __init__(self, acoustic_env, channel_to_model=0, given_excitation=None,
                 use_matlab=False,
                 transition_time=0.024, early_part_only=False, samples_per_opt_span_early=1,
                 offset_to_max=False, use_caches=False, plot_exci=False,
                 quiet=False, enforce_dp_sanity_check=True,
                 ac_params={}):
        """

        Model initialization. To run the modeling do:
        acenvmodel.get_model()

        Args:
            acoustic_env: An instance of AcEnv. This contains the base acoustic environment to be
                            modeled
            channel_to_model: The channel of the original AIR to model
            given_excitation: Given excitation samples
            use_matlab: Flag for using MATLAB engine
            transition_time: The transition time between the early and late parts, measured from
                            the ToA of the direct sound
            early_part_only: Flag for modeling only the early part
            samples_per_opt_span_early: Samples per optimization step of the early part
            offset_to_max: Remove leading silence in AIR
            use_caches: Flag for the use of caches
            plot_exci: Plotting flag
            quiet: Quiet processing
            enforce_dp_sanity_check: Check if the direct sound assumptions are valid
            ac_params: A dictionary of provided acoustic parameters

        """

        self.name = acoustic_env.name
        self.plot_exci = plot_exci
        self.quiet = quiet
        # Early Part Settings
        self.early_opt_max_ref = 1
        self.early_opt_delayresolution = 0.05
        self.early_opt_opt_span = samples_per_opt_span_early / self.early_opt_delayresolution
        self.lasso_esigma_lambda = 1 / 4.
        self.soa_decimal_precision = 4
        self.early_opt_spars_prom_factor = 0.90
        self.edc_compensation_enable = True
        self.single_side_dir_path_winlen = 0.0002
        self.tail_visible_energy_ratio = 1
        self.edc_scale_lim = 0.01
        self.direct_sound_neighborhood = .0003
        self.use_caches = use_caches
        self.cache_dir = '../results_dir/acenvgenmodel_cachedir/git_' + get_git_hash() + '/'
        # Early Part Params
        self.lasso_b = np.array([])
        self.paramset_directsound_soa = np.array([])
        self.paramset_directsound_amp = np.array([])
        self.paramset_reflections_fval_reduction = np.array([])
        self.paramset_reflections_soa = np.array([])
        self.paramset_reflections_amp = np.array([])
        self.paramset_excitation_params = np.array([])
        self.edc_scale = None
        # Late Part Settings
        self.model_late_part = not early_part_only
        self.late_opt_delayresolution = 1 / 2.
        self.late_tail_gaussian_std = 10
        self.late_num_ma_coeff = 5
        self.late_num_ar_coeff = 5
        self.late_proc_framelength = 512
        self.use_matlab = use_matlab
        # Late Part Params
        self.paramset_tail_modes_ar_coeff = np.array([])
        self.paramset_tail_modes_ma_coeff = np.array([])
        # Model Settings
        self.defined_transition_time = transition_time
        self.use_delta_as_excitation = acoustic_env.is_simulation
        self.given_excitation = given_excitation
        self.early_only = True
        # Model Parameters
        self.printer('Getting channel ' + str(channel_to_model) + ' from the AIR')
        self.original = np.array(acoustic_env.impulse_response[:, channel_to_model])
        self.original = self.original / abs(self.original).max()
        self.sampling_freq = acoustic_env.sampling_freq
        self.paramset_reverberation_time = 0
        self.paramset_drr_linscale = 0
        self.paramset_drr_linscale_early = 0
        self.transition_sample = 0
        # Acoustic environment specification
        self.acoustic_env = acoustic_env
        self.channel_to_model = channel_to_model
        # Result
        self.air_reconstructed_from_model = np.array([])
        self.number_of_parameters = np.nan

        self.original = self.original / abs(self.original).max()
        if offset_to_max:
            backsamples = int(np.floor(0.005 * self.sampling_freq))
            self.original = self.original[max(0, abs(self.original).argmax() - backsamples):]

        self.transition_sample = abs(self.original).argmax() + \
                                 int(np.ceil(self.sampling_freq * self.defined_transition_time))

        if 't60' in ac_params:
            self.paramset_reverberation_time = ac_params['t60']
        else:
            try:
                self.paramset_reverberation_time = get_t60_decaymodel(np.array(self.original),
                                                                      self.sampling_freq)
                self.printer(
                    getfname() + " : T60 estimated as : " + float2str(
                        self.paramset_reverberation_time) + 's')
            except ValueError:
                self.paramset_reverberation_time = np.nan

        if 'drr_early' in ac_params:
            self.paramset_drr_linscale_early = ac_params['drr_early']
        else:
            self.paramset_drr_linscale_early = get_drr_linscale(
                self.original[0:self.transition_sample],
                self.sampling_freq)

        if 'drr_late' in ac_params:
            self.paramset_drr_linscale_late = ac_params['drr_late']
        else:
            self.paramset_drr_linscale_late = get_drr_linscale(
                np.array(self.original),
                self.sampling_freq,
                ignore_reflections_up_to=self.defined_transition_time)

        if 'drr' in ac_params:
            self.paramset_drr_linscale = ac_params['drr']
        else:
            self.paramset_drr_linscale = get_drr_linscale(np.array(self.original),
                                                          self.sampling_freq)

        self.printer(
            getfname() + " : DRR estimated as : " + float2str(
                10 * np.log10(self.paramset_drr_linscale)) + "dB")
        self.printer(
            getfname() + " : DRR early estimated as : " + float2str(
                10 * np.log10(self.paramset_drr_linscale_early)) + "dB")
        self.printer(
            getfname() + " : DRR late estimated as : " + float2str(
                10 * np.log10(self.paramset_drr_linscale_late)) + "dB")

        if enforce_dp_sanity_check:
            try:
                self.direct_path_sanity_check()
            except AssertionError as ME:
                print('Sanity check failed for the direct sound of ' +
                      self.name + ' but you said you don\'t want to stop. '
                                  'Message: ' + str(ME.message))

        run_command('mkdir -p ' + self.cache_dir)

    def printer(self, statement, force_loud=False):
        """
        Prints a string on stdout. Skips the printing if operating in quiet mode.

        Args:
            statement: The string to print
            force_loud: Force output

        Returns:
            Nothing

        """
        if (not self.quiet) or force_loud:
            print(statement)

    def direct_path_sanity_check(self):
        """
        Empirically checks whether the assumtpions made about the direct path sound are valid.

        Returns:
            Nothing

        """
        air = np.array(self.original)
        air /= max(abs(air))
        enprop = np.cumsum(air ** 2) / np.sum(air ** 2)
        max_sample = np.argmax(abs(air))
        sanity_coef = np.sum(enprop[0:max_sample - 1] ** 2)
        if sanity_coef > 0.01:
            raise AssertionError('I refuse to model this because i do not think that my direct '
                                 'sound assumption is valid')
        else:
            self.printer('I think the direct sound is ok...')

    def get_model(self, make_plots=False):
        """
        Performs the main modeling routine

        Args:
            make_plots: Make plots of intermidiate results.

        Returns:
            The AIR reconstruction

        """

        self.early_lasso_lin_approx()
        if make_plots:
            self.plot_modeling_results(interactive=True, suptitle='LASSO Lin. Approx. Results',
                                       reconstruct=True)
        if self.plot_exci:
            plot_matched_excitation(self.original[0:self.transition_sample],
                                    self.excitation_model_reconstruct()[0])
        self.early_lasso_postproc()

        if self.model_late_part:
            self.late_part_find_room_modes_ar_coeff()
            self.late_tail_generation()

        self.printer(getfname() + ' : Done')
        if make_plots:
            self.plot_modeling_results(interactive=True, suptitle='Final Results', reconstruct=True)

        return self.air_reconstruct()

    def get_edc_scale(self, up_to=None, reference=None):
        """
        Get the scaling vector to use for the compensation of the energy decay based on the EDC

        Args:
            up_to: Level up to which to get down to
            reference: The signal to use for the scale estimation. If None, then the environments
                        AIR will be used

        Returns:
            The scale

        """
        if reference is None:
            reference = np.array(self.original)
        if self.edc_compensation_enable:
            self.printer(getfname() + ' : EDC Compensation')
            return get_edc_scale(reference, up_to=up_to, edc_scale_lim=self.edc_scale_lim)
        else:
            raise AssertionError('EDC compensation is disabled')

    def early_lasso_lin_approx(self):
        """
        The LASSO is used to approximate the non-linear problem of refection detection. It is
        used to initialize the modeling of the ealry part of the AIR. The method updates the
        members of the object.

        Returns:
            Nothing

        """
        reference = self.original / abs(self.original).max()
        if self.edc_compensation_enable:
            this_edc_scale, original_edc_scaled = self.get_edc_scale(up_to=self.transition_sample,
                                                                     reference=reference)
            self.edc_scale = this_edc_scale
        else:
            self.printer(getfname() + ' : Skipping EDC Compensation')
            self.edc_scale = np.ones(reference.shape)
            original_edc_scaled = np.array(reference)
            this_edc_scale = np.ones((self.transition_sample,))

        original_edc_scaled = original_edc_scaled[0:self.transition_sample]
        original_early = reference[0:self.transition_sample]

        if self.given_excitation is not None:
            excitation, _ = self.excitation_model_reconstruct()
            self.printer(
                getfname() + ' : Skipping Excitation Estimation, using given excitation of '
                             'shape ' + str(excitation.shape))
        elif not self.use_delta_as_excitation:
            self.printer(getfname() + ' : Excitation Estimation')

            cache_file = self.cache_dir + '/excitation_params_' + self.name + '.pkl'
            if self.use_caches and os.path.isfile(cache_file):
                self.printer('Using cache file for excitation_params ' + cache_file)
                with open(cache_file) as pklf:
                    self.paramset_excitation_params = load(pklf)
            else:

                drrsidelength_samples = np.ceil(
                    self.single_side_dir_path_winlen * self.sampling_freq)
                _, maxenpoint = matmax(abs(original_early))
                dpathobserved = original_early[int(max(1, maxenpoint - drrsidelength_samples)):
                                               int(maxenpoint + drrsidelength_samples)]
                excitation_period = np.mean((1 / 20e3, 1 / (0.1 * self.sampling_freq)))
                # x0 = row_vector([excitation_period, excitation_period])
                opt_lb = (1 / 20e3, 1E-10, -np.pi / 2.)
                opt_ub = (1 / (0.1 * self.sampling_freq), 0.5 * excitation_period, +np.pi / 2.)

                def opt_fun(x):
                    return self.excitation_model_mse_func(self.excitation_model(x[0], x[1], x[2]),
                                                          dpathobserved)

                self.paramset_excitation_params = brute(opt_fun, (
                    (opt_lb[0], opt_ub[0]),
                    (opt_lb[1], opt_ub[1]),
                    (opt_lb[2], opt_ub[2])), Ns=20, finish=fmin)
                # self.excitation_model_mse_func(excitation, dpathobserved, doplot=1)

                self.printer('Storing excitation_paras results' + cache_file)
                with open(cache_file, 'w') as pklf:
                    dump(self.paramset_excitation_params, pklf)
            self.printer('Excitation params:' + str(self.paramset_excitation_params))
            excitation = self.excitation_model_reconstruct()[0]

        else:
            excitation = np.array([])
            self.printer(getfname() + ' : Skipping Excitation Estimation, using delta')

        self.printer(getfname() + ' : Estimating Linear Models')
        delays = np.arange(0, original_early.size, self.early_opt_delayresolution)

        pointbasescale = self.edc_scale[np.array(np.floor([delays[0]] + delays.tolist()[0:-1]
                                                          ).tolist(), dtype=np.int)]
        linear_terms = gm_frac_delayed_copies(np.ones(delays.shape), delays,
                                              original_early.size, row_vector(excitation))[1]
        linear_terms *= 1.0 / abs(linear_terms).max()

        use_matlab = self.use_matlab
        if use_matlab:
            try:
                import matlab.engine
            except ImportError:
                self.printer(getfname() + " : Cannot import MATLAB, will use ElasticNet. This "
                                          "will take a long time. If you have Matlab, you can set "
                                          "it up to work with Python: "
                                          "https://uk.mathworks.com/help/matlab/"
                                          "matlab-engine-for-python.html")
                use_matlab = False

        if self.edc_compensation_enable:
            linear_term_skewed = linear_terms / np.repeat(column_vector(this_edc_scale),
                                                          linear_terms.shape[1], axis=1)
        else:
            linear_term_skewed = linear_terms

        cache_file = self.cache_dir + '/lasso_' + self.name + '.pkl'
        if self.use_caches and os.path.isfile(cache_file):
            self.printer('Using cache file for lasso ' + cache_file)
            with open(cache_file) as pklf:
                lamdas_lasso, coefs_lasso = load(pklf)
        else:
            if not self.use_caches:
                self.printer('I was told not to use caches so i did not look for ' + cache_file)
            else:
                self.printer('I did not find the file ' + cache_file)
            if use_matlab:
                self.printer(getfname() + ' : Starting MATLAB')
                eng = matlab.engine.start_matlab()
                self.printer(getfname() + ' : Converting data to MATLAB format')
                predictors = matlab.double(linear_term_skewed.tolist())
                indipendent_variable = matlab.double(column_vector(original_edc_scaled).tolist())
                self.printer(getfname() + ' : Evaluating MATLAB LASSO')
                coefs_lasso, matlab_lasso_fit_info = eng.lasso(predictors, indipendent_variable,
                                                               'NumLambda', 20.0, nargout=2)
                eng.exit()
                self.printer(getfname() + ' : Getting results from MATLAB')
                lamdas_lasso = np.array(matlab_lasso_fit_info['Lambda'])[0]
            else:
                lamdas_lasso, coefs_lasso, _ = ElasticNet.path(linear_term_skewed, column_vector(
                    original_edc_scaled), eps=1e-3, l1_ratio=1, n_alphas=20, copy_X=True,
                                                               precompute=True)  # 100
                coefs_lasso = coefs_lasso[0]
                coefs_lasso = coefs_lasso[:, ::-1]
                lamdas_lasso = lamdas_lasso[::-1]
            self.printer('Storing lasso results' + cache_file)
            with open(cache_file, 'w') as pklf:
                dump([lamdas_lasso, coefs_lasso], pklf)

        self.printer(getfname() + ' : Choosing Linear Model')
        coefs_lasso = np.array(coefs_lasso)
        lasso_approximations = np.matmul(coefs_lasso.transpose(),
                                         linear_term_skewed.transpose()).transpose()
        lasso_approximations_mse = np.sum(
            np.square(lasso_approximations -
                      np.repeat(column_vector(original_edc_scaled), lamdas_lasso.size, axis=1)),
            axis=0)
        mimse, minmse_idx = matmin(lasso_approximations_mse.flatten())
        if self.lasso_esigma_lambda == 0:
            lambda1stde = minmse_idx
        else:
            lambda1stde = minmse_idx + \
                          np.where(lasso_approximations_mse[minmse_idx:] <
                                   mimse +
                                   (np.std(lasso_approximations_mse) * self.lasso_esigma_lambda)
                                   )[-1][-1]
        lasso_model_orders = np.sum(coefs_lasso != 0, axis=0)
        if lasso_model_orders[lambda1stde] == 0:
            self.printer('Zero-order model chosen by LASSO... will continue')
        coefs_lasso *= np.repeat(column_vector(pointbasescale), lamdas_lasso.size, axis=1)
        coefs_lasso = coefs_lasso[:, int(lambda1stde)]
        self.lasso_b = np.array(coefs_lasso)
        self.printer(getfname() + ' : Choosing Model #' + str(lambda1stde) + ' of ' +
                     str(lamdas_lasso.size) + ' with ' +
                     str(lasso_model_orders[lambda1stde]) + ' nonzero values at lambda ' +
                     str(self.lasso_esigma_lambda))

        nzcoefs = np.array(np.where(self.lasso_b != 0))[0]

        if nzcoefs.size == 0:
            self.paramset_directsound_soa = np.array([])
            self.paramset_directsound_amp = np.array([])
            self.paramset_reflections_soa = np.array([])
            self.paramset_reflections_amp = np.array([])
        else:
            self.paramset_directsound_soa = nzcoefs[0] * self.early_opt_delayresolution
            self.paramset_directsound_amp = self.lasso_b[nzcoefs[0]]
            self.paramset_reflections_soa = nzcoefs[1:] * self.early_opt_delayresolution
            self.paramset_reflections_amp = self.lasso_b[nzcoefs[1:]]

        self.paramset_reflections_fval_reduction = np.zeros_like(self.paramset_reflections_soa)

    def excitation_model(self, cosine_period, gaussian_sigma, phase_shift):
        """

         Realisation of a modulated Gaussian pulse given the parameters

        Args:
            cosine_period: Modulation cosine period
            gaussian_sigma: The std of the Gaussian Component
            phase_shift: The phase offset of the cosine

        Returns:
            The realisation of a modulated Gaussian pulse formed by the input parameters
        """
        drrsidelength_samples = np.ceil(self.single_side_dir_path_winlen * self.sampling_freq)
        sample_index = np.arange(1, 2 * drrsidelength_samples) - drrsidelength_samples
        shaping = norm(loc=0, scale=gaussian_sigma).pdf(sample_index / self.sampling_freq) / \
                  self.sampling_freq
        shaping /= max(abs(shaping))
        gpulse = shaping * np.cos(phase_shift +
                                  2 * np.pi / cosine_period * sample_index / self.sampling_freq)
        maxval = abs(gpulse).max()
        if maxval > 0:
            gpulse = gpulse / maxval
        return gpulse

    def excitation_model_reconstruct(self):
        """

         Returns the excitation for the AIR

        Args:
            Nothing

        Returns:
            The AIR estimated excitation
        """

        if self.use_delta_as_excitation:
            nparams = 0
            return np.array([]), nparams

        if self.given_excitation is not None:
            excitation = np.array(self.given_excitation)
            nparams = excitation.size
        else:
            excitation = self.excitation_model(self.paramset_excitation_params[0],
                                               self.paramset_excitation_params[1],
                                               self.paramset_excitation_params[2])
            nparams = self.paramset_excitation_params.size

        excitation /= abs(excitation).max()
        if excitation[abs(excitation).argmax()] < 0:
            excitation *= -1

        return excitation, nparams

    def excitation_model_mse_func(self, exci_estimate, exci_observed, doplot=0):
        """
        Returns the MSE between a near-to-optimally shifted version of exci_estimate and
        exci_observed.

        Args:
            exci_estimate: The vector containing an approximation for the direct path sound
            exci_observed: A window believed to contain the direct path sound
            doplot: Boolean that defines if the result should be plotted after alignment

        Returns:
            The MSE between a near-to-optimally shifted version of exci_estimate and
        exci_observed.

        """
        fracdelayresolution = self.early_opt_delayresolution  # / 10
        delays = np.arange(0, exci_observed.size - fracdelayresolution, fracdelayresolution)
        shiftedcopies = gm_frac_delayed_copies(np.ones(delays.shape), delays, exci_observed.size,
                                               center_filter_peaks=True,
                                               excitation_signal=row_vector(
                                                   exci_estimate))[1]
        msevals = np.sum((np.repeat(column_vector(exci_observed), delays.size, axis=1) -
                          shiftedcopies) ** 2, axis=0)
        delaymatch = msevals.argmin()
        mse = msevals[delaymatch]

        if doplot == 1:
            import matplotlib.pyplot as mplot
            mplot.plot(exci_observed, label='Observed Window')
            mplot.plot(shiftedcopies[:, delaymatch], label='Estimated Excitation')
            mplot.legend(loc='upper right', fontsize=8)
            mplot.show()

        return mse

    def early_lasso_postproc_opt(self, candidate_amps, candidate_dels, num_reflections, max_amp,
                                 back_shift, front_shift, air_samples, current_reconstruction,
                                 verbose=False, ):
        """
        Fine-tunes the ToA and Amplitude for a set of reflections, in a window of the AIR.

        Args:
            candidate_amps: The candidate reflections
            candidate_dels: The candidate ToAs
            num_reflections: The number of reflections to consider
            max_amp: The maximum amplitude to consider
            back_shift: The minumum ToA to consider
            front_shift: The maximum ToA to consider
            air_samples: The samples of the AIR window
            current_reconstruction: The current recosntruction of the window
            verbose: Verbose reporting

        Returns:
            The amplitudes of the reflections
            The ToAs of the reflections
            The reduction in the MSE of the error by the given reflections

        """

        if isinstance(num_reflections, list):

            for i in range(1, len(num_reflections)):
                if not num_reflections[i] - num_reflections[i - 1] == 1:
                    raise AssertionError('Expected sequential reflection counts')

            amplitudes = [None] * len(num_reflections)
            delays = [None] * len(num_reflections)
            fval = [None] * len(num_reflections)
            for i in range(len(num_reflections)):
                amplitudes[i], delays[i], fval[i] = \
                    self.early_lasso_postproc_opt(candidate_amps, candidate_dels,
                                                  num_reflections[i],
                                                  max_amp,
                                                  back_shift, front_shift, air_samples,
                                                  current_reconstruction
                                                  )
        else:

            excitation = row_vector(self.excitation_model_reconstruct()[0])

            def optfis(x):
                x = np.array(x)
                local_amplitudes = x[0:num_reflections]
                local_delays = x[num_reflections:2 * num_reflections]

                this_and_prev = current_reconstruction + \
                                self.early_part_reconstruct(desired_length=air_samples.size,
                                                            amplitudes=local_amplitudes,
                                                            delays=local_delays,
                                                            amplitudes_direct=[],
                                                            delays_direct=[],
                                                            excitation=excitation,
                                                            no_rescaling=True,
                                                            assert_on_0_energy=False)[0]

                error_measure = np.sum((air_samples - this_and_prev) ** 2)

                if np.any(np.isnan(error_measure)):
                    raise ValueError('Found NaN in MSE calculation')
                return error_measure
 
            bound_amps = np.repeat([[-5, +5]], num_reflections, axis=0)
            bound_dels = []
            epsilon = 10. ** (-self.soa_decimal_precision)
            for deli in candidate_dels:
                if deli % 1 == .5:
                    bound_dels.append((np.floor(deli), np.ceil(deli)))
                elif deli % 1 < .5:
                    bound_dels.append((np.floor(deli) - .5, np.ceil(deli) - .5 - epsilon))
                else:
                    bound_dels.append((np.floor(deli) + .5, np.ceil(deli) + .5 - epsilon))
            x0_amp = candidate_amps[0:num_reflections]
            x0_del = candidate_dels[0: num_reflections]
            x0 = np.concatenate((x0_amp, x0_del))

            bound_tuple = (bound_amps, bound_dels)
            bnds = tuple(map(tuple, np.concatenate(bound_tuple)))

            # Original paper proposes interior point. Here we rely on minimize to pick an algorithm
            optres = minimize(optfis, x0, bounds=bnds)
            outp = optres.x
            fval = optres.fun

            amplitudes = np.array(outp[0:num_reflections])
            delays = np.array(outp[num_reflections:2 * num_reflections])

            if verbose:
                self.printer('For candidate delays: ' + float2str(candidate_dels) + ' and ' +
                             'for candidate amps :' + float2str(candidate_amps))
                self.printer('Estimated delays: ' + float2str(delays) + ' and ' +
                             'estimated amps: ' + float2str(amplitudes))

                import matplotlib.pyplot as plt

                reco = self.early_part_reconstruct(desired_length=air_samples.size,
                                                   amplitudes=amplitudes,
                                                   delays=delays,
                                                   amplitudes_direct=[],
                                                   delays_direct=[],
                                                   excitation=excitation,
                                                   no_rescaling=True,
                                                   assert_on_0_energy=False)[0]

                plt.clf()
                plt.plot(air_samples, label='Original', linewidth=.5, marker='o')
                plt.plot(current_reconstruction, label='Current', linewidth=.5, marker='.')
                plt.plot(reco, label='New', linewidth=.5, marker='+')
                plt.grid(True)
                plt.legend(fontsize=8)
                plt.show()

        return amplitudes, delays, fval

    def early_lasso_postproc(self, verbose=False, scan_order='forward'):
        """
        Performs the post-processing of the LASSO regression coefficients to extract the
        reflections' Sample of Arrival and Amplitudes. The LASSO is used to approximate the
        non-linear problem of refection detection.

        Examples:
            Proceed the call to this function by:
                acoustic_env_model.early_lasso_lin_approx()
            Then to post-process the results:
                acoustic_env_model.early_lasso_postproc()

        Args:
            verbose: Verbose reporting
            scan_order: Order in which to do the reflection modeling, foward starts fro mthe
            first reflection in time and moves later in time. Backwards starts the modeling from
            the last reflection and moves backwards to the first.

        Returns:
            Nothing

        """

        from utils_spaudio import fractional_alignment

        air_postproc_region = np.array(self.original[0:self.transition_sample])

        self.printer(getfname() + ' : Post-processing LASSO betas')
        ndp = self.soa_decimal_precision
        cap_reflections_to = self.early_opt_max_ref
        opt_span = self.early_opt_opt_span
        sparsity_promoting_factor_base = self.early_opt_spars_prom_factor

        def truncatethis(x):
            trout = np.atleast_1d((x * (10. ** ndp)).round() / (10. ** ndp))
            return trout

        local_area_in_seconds_singlesided = 0.0005 * self.sampling_freq
        lareass = np.ceil(local_area_in_seconds_singlesided)
        delayresolution = self.early_opt_delayresolution

        newb2 = self.lasso_b.flatten()
        n_samples_front_pad = float(opt_span) / 2.
        if not n_samples_front_pad % 1 == 0:
            raise AssertionError('Cannot offset by haf sample if the number of predictors within'
                                 ' a sample soan is not even')
        n_samples_front_pad = int(n_samples_front_pad)
        newb2 = np.concatenate((np.zeros(n_samples_front_pad, dtype=newb2.dtype), newb2))
        newb2_enframed = enframe(newb2, int(opt_span), int(opt_span))

        hd = self.excitation_model_reconstruct()[0]

        self.printer(getfname() + ' : Preparing direct sound')

        max_en_original = abs(self.original).argmax()
        if hd.size == 0:
            dspan = 16
        else:
            dspan = hd.size
        early_region_offset = max(0, max_en_original - 2 * dspan)
        early_region = self.original[early_region_offset:
                                     min(self.original.size, max_en_original + 1 * dspan)]
        reconstruct_early = gm_frac_delayed_copies(
            [1], max_en_original - early_region_offset,
            early_region.size, np.atleast_2d(hd))[0]

        aligned_dpath, dp_ls_delay, dp_ls_amp = fractional_alignment(
            np.concatenate((np.atleast_2d(early_region), np.atleast_2d(reconstruct_early)), axis=0),
            ls_scale=True, resolution=0.0001)

        self.paramset_directsound_soa = max_en_original + dp_ls_delay[1]
        self.paramset_directsound_amp = dp_ls_amp[1]

        directsound_toa = self.paramset_directsound_soa / float(self.sampling_freq)

        delay_lookup_table = np.reshape(
            np.arange(0, newb2_enframed.size * self.early_opt_delayresolution,
                      self.early_opt_delayresolution),
            newb2_enframed.shape) - np.float(.5)
        newb2_enframed[delay_lookup_table[:] < directsound_toa + self.direct_sound_neighborhood] = 0
        active_b_frames = np.array(np.sum(newb2_enframed, axis=1).nonzero()).flatten()

        bestdelays = np.zeros((active_b_frames.size * cap_reflections_to + 1,))
        best_fval_reductions = np.zeros((active_b_frames.size * cap_reflections_to + 1,))
        bestamplitudes = np.zeros((active_b_frames.size * cap_reflections_to + 1,))

        best_fval_reductions[0] = 0
        bestdelays[0] = self.paramset_directsound_soa
        bestamplitudes[0] = self.paramset_directsound_amp
        total_reflection_counter = 1

        sign_flip = -1
        if scan_order == 'backward':
            sign_flip = 1
        elif not scan_order == 'forward':
            raise AssertionError('Invalid option for modeling order')
        active_b_frames = active_b_frames[
            (sign_flip * np.sum(abs(newb2_enframed[active_b_frames, :]), axis=1)).argsort()]

        acceptednrefs_reconstruct = self.early_part_reconstruct(
            amplitudes=np.array([bestamplitudes[0]]),
            delays=np.array([bestdelays[0]]), desired_length=air_postproc_region.size,
            excitation=row_vector(hd),
            amplitudes_direct=[], delays_direct=[], no_rescaling=True)[0]

        self.printer(getfname() + ' : Post-processing LASSO betas : Optimize')

        self.printer(getfname() + ' : To process : ' +
                     str(active_b_frames.size) + ' groups')
        group_counter = 0

        for current_frame_in_b in active_b_frames.tolist():

            group_counter += 1
            if group_counter % 10 == 1:
                self.printer(
                    getfname() + ' : Processing group : ' + str(group_counter) + ' out of : ' +
                    str(active_b_frames.size) + ' groups')
            samewindow = (-abs(newb2_enframed[current_frame_in_b, :])).argsort()[
                         0:cap_reflections_to]
            samewindow_delays = delay_lookup_table[current_frame_in_b, samewindow].flatten()
            region_start = np.floor(current_frame_in_b * opt_span * delayresolution - .5)
            backshift = min(region_start, lareass)
            local_delay_shift = region_start - backshift
            samewindow_amplitudes = newb2_enframed[current_frame_in_b, samewindow].flatten()
            ldtotal = samewindow_delays.size

            region_end = np.ceil(current_frame_in_b * opt_span * delayresolution + .5)
            frontshift = min(air_postproc_region.size - region_end, lareass)
            localy = air_postproc_region[int(local_delay_shift): int(region_end + frontshift)]
            local_acceptednrefs_reconstruct = acceptednrefs_reconstruct[int(
                local_delay_shift): int(region_end + frontshift)]

            if (localy == 0).all():
                continue

            samewindow_delays_shifted = samewindow_delays - local_delay_shift
            zeroreffval = sum((localy - local_acceptednrefs_reconstruct) ** 2)
            orderbyhighest = np.argsort(-abs(samewindow_amplitudes))
            samewindow_amplitudes = samewindow_amplitudes[orderbyhighest]
            samewindow_delays_shifted = samewindow_delays_shifted[orderbyhighest]
            acceptednrefs = 0
            bestdelays_local = np.array([])
            best_fval_reductions_local = np.array([])
            bestamplitudes_local = np.array([])
            maxlocalamp = max(abs(localy))

            theseamplitudes_all, thesedelays_all, fval = \
                self.early_lasso_postproc_opt(samewindow_amplitudes,
                                              samewindow_delays_shifted,
                                              list(range(1, ldtotal + 1)),
                                              maxlocalamp, backshift,
                                              frontshift,
                                              localy,
                                              local_acceptednrefs_reconstruct)

            for i in range(1, ldtotal + 1):

                theseamplitudes = theseamplitudes_all[i - 1]
                thesedelays = truncatethis(thesedelays_all[i - 1])
                if i == 1:
                    prevscore = zeroreffval
                else:
                    prevscore = fval[i - 2]
                these_fval_reductions = prevscore - fval[i - 1]

                weighted_score = fval[i - 1] / sparsity_promoting_factor_base
                if prevscore > weighted_score:
                    if verbose:
                        self.printer('Accepted to add reflection ' + str(i) + ' to window')
                        self.printer(
                            'Current score is ' + float2str(weighted_score) + ' and prev is '
                            + float2str(prevscore))
                    acceptednrefs = np.unique(thesedelays).size
                    amplitudes = np.array([])
                    delays = np.array([])
                    fval_reductions = np.array([])
                    while thesedelays.size > 0:
                        delays = np.append(delays, thesedelays[0] + local_delay_shift)
                        amplitudes = np.append(amplitudes,
                                               sum(theseamplitudes[thesedelays == thesedelays[0]]))
                        fval_reductions = np.append(fval_reductions,
                                                    [these_fval_reductions] * delays.size)
                        keepthese = np.logical_not(thesedelays == thesedelays[0])
                        theseamplitudes = np.atleast_1d(theseamplitudes[keepthese])
                        thesedelays = np.atleast_1d(thesedelays[keepthese])
                    bestdelays_local = delays
                    bestamplitudes_local = amplitudes
                    best_fval_reductions_local = fval_reductions
                else:
                    if verbose:
                        self.printer('Rejected to add reflection ' + str(i) + ' to window')
                        self.printer(
                            'Current score is ' + float2str(weighted_score) + ' and prev is '
                                                                              '' + float2str(
                                prevscore))
                if verbose:
                    self.printer(
                        'Currently accepted ' + str(total_reflection_counter) + ' reflections')
            if acceptednrefs > 0:
                writerange = total_reflection_counter + np.arange(acceptednrefs)
                bestdelays[writerange] = bestdelays_local
                best_fval_reductions[writerange] = best_fval_reductions_local
                bestamplitudes[writerange] = bestamplitudes_local
                total_reflection_counter += acceptednrefs
                acceptednrefs_reconstruct += self.early_part_reconstruct(
                    amplitudes=bestamplitudes_local,
                    delays=bestdelays_local, desired_length=air_postproc_region.size,
                    excitation=row_vector(hd),
                    amplitudes_direct=[], delays_direct=[], no_rescaling=True)[0]

        bestdelays = bestdelays[0:total_reflection_counter]
        bestamplitudes = bestamplitudes[0:total_reflection_counter]
        best_fval_reductions = best_fval_reductions[0:total_reflection_counter]
        self.printer(getfname() + ' : Done with post-processing LASSO betas ,processed : ' +
                     str(active_b_frames.size) + ' groups')

        if total_reflection_counter > 0:
            bestamplitudes = bestamplitudes[bestdelays.argsort()]
            bestdelays = bestdelays[bestdelays.argsort()]
            best_fval_reductions = best_fval_reductions[bestdelays.argsort()]
            max_en_original = abs(self.original).argmax()

            direct_reach = max_en_original + np.ceil(
                self.direct_sound_neighborhood * self.sampling_freq).astype(int)
            direct_samples = np.where(bestdelays < direct_reach)[-1]

            self.paramset_reflections_soa = bestdelays[direct_samples.max() + 1:]
            self.paramset_reflections_amp = bestamplitudes[direct_samples.max() + 1:]
            self.paramset_reflections_fval_reduction = best_fval_reductions[
                                                       direct_samples.max() + 1:]
        else:
            raise AssertionError("No reflections found")

    def early_part_reconstruct(self,
                               desired_length=None,
                               amplitudes=None,
                               delays=None,
                               amplitudes_direct=None,
                               delays_direct=None,
                               excitation=None,
                               no_rescaling=True,
                               assert_on_0_energy=False):
        """
        Reconstructs teh FIR filter taps that represent the early part of the AIR being modeled.

        Args:
            desired_length: The length of the reconstruction
            amplitudes: The amplitudes of the reflections
            delays: The ToAs of the reflections
            amplitudes_direct: The amplitude of the direct sound
            delays_direct: The ToA of the direct sound
            excitation: The excitation for the AIR measurement
            no_rescaling: Skip the normalization of the scales
            assert_on_0_energy: Throw an AssertionError when the modeling produces no energy

        Returns:
            The earlt part reconstruction
            The direct sound reconstruction
            The number of parameters needed to model this

        """

        nones = [
            amplitudes is None,
            delays is None,
            amplitudes_direct is None,
            delays_direct is None,
            excitation is None]
        if np.any(nones) and not np.all(nones):
            raise AssertionError(
                'If any reflection or direct sound parameter is specified '
                'it needs to be accompanied by all other parameters')

        if desired_length is None:
            desired_length = self.transition_sample
        if amplitudes is None:
            amplitudes = self.paramset_reflections_amp
        if delays is None:
            delays = self.paramset_reflections_soa
        if amplitudes_direct is None:
            amplitudes_direct = self.paramset_directsound_amp
        if delays_direct is None:
            delays_direct = self.paramset_directsound_soa
        if excitation is None:
            excitation, nparams_exci = self.excitation_model_reconstruct()
            excitation = row_vector(excitation)
        else:
            nparams_exci = np.nan
        if np.array(amplitudes_direct).size == 0:
            no_rescaling = True

        nparams = amplitudes.size + \
                  delays.size + \
                  np.array(amplitudes_direct).size + \
                  np.array(delays_direct).size + \
                  nparams_exci

        est_direct = gm_frac_delayed_copies(
            amplitudes_direct,
            delays_direct,
            desired_length,
            excitation,
            True)[0]

        est_early_rest = gm_frac_delayed_copies(amplitudes, delays, desired_length,
                                                excitation, True)[0]

        if no_rescaling:
            est_early_all = est_direct + est_early_rest
        else:
            scale = abs(est_direct).astype(float).max()
            est_early_all = est_direct / scale + est_early_rest / scale

        if np.any(np.isnan(est_early_all)):
            raise ValueError('Found NaNs in reconstruction')

        return est_early_all, est_direct, nparams

    def early_part_plot_result(self):
        """

        Plots a summary of the fitting results for the early part of the AIR

        Returns:
            Nothing

        """

        import matplotlib.pyplot as mplot

        t = np.arange(0, self.transition_sample) / float(self.sampling_freq)
        mplot.plot(t, self.original[0:self.transition_sample], label='AIR', linewidth=0.5)
        mplot.plot(t, self.early_part_reconstruct(no_rescaling=True)[0].flatten(),
                   label='Reconstructed',
                   linewidth=0.5)
        mplot.plot(self.paramset_directsound_soa / float(self.sampling_freq), 0, 'r+',
                   label='Est. Dir. Sound')
        mplot.plot(self.paramset_reflections_soa / (self.sampling_freq), np.zeros(
            self.paramset_reflections_soa.size), 'k+',
                   label='Est. Reflections')
        mplot.xlabel('Time (s)')
        mplot.title('Early Part Fit Results')
        mplot.legend(fontsize=8)
        mplot.grid()
        mplot.show()

    def late_part_find_room_modes_ar_coeff(self, plot_ar_impulse_response=False):
        """
        Finds AR coefficients that will filter the tail of the AIR to model the room modes.

        Examples:
            acoustic_env_model.late_part_find_room_modes_ar_coeff()

        Args:
            plot_ar_impulse_response: Plots the modeling result

        Returns:
            Nothing

        """

        self.printer(getfname() + " : Finding AR coefficients to model room modes")
        late_ir_analysis = air_up_to_db(np.array(self.original), -30)
        late_ir_analysis = late_ir_analysis[self.transition_sample:]
        # late_ir_analysis *= np.hamming(late_ir_analysis.size)
        max_energy_sample_late = abs(late_ir_analysis).argmax()
        if late_ir_analysis[max_energy_sample_late] < 0:
            late_ir_analysis = -late_ir_analysis
        if late_ir_analysis.size - max_energy_sample_late < self.late_num_ar_coeff:
            raise NameError(getfname() + ':MaxEnergySampleTooLate')
        late_ir_analysis = late_ir_analysis[max_energy_sample_late:]

        if self.late_num_ma_coeff == 0:
            self.printer(
                'I will not look at any MA representation and will just get an AR model using LPC')
            ar_mdl = lpc(late_ir_analysis, order=self.late_num_ar_coeff)
            self.paramset_tail_modes_ar_coeff = ar_mdl.numerator
            ma_mdl = [1]
            self.paramset_tail_modes_ma_coeff = ma_mdl
        else:
            self.printer(
                'You want to include AR and MA coefficients so i will use Prony\'s method '
                'to get both')
            ma_mdl, ar_mdl = prony(late_ir_analysis, self.late_num_ma_coeff, self.late_num_ar_coeff)
            self.paramset_tail_modes_ar_coeff = ar_mdl
            self.paramset_tail_modes_ma_coeff = ma_mdl

        if plot_ar_impulse_response:
            impulse = np.zeros((int(np.ceil(0.005 * self.sampling_freq)),))
            impulse[0] = 1
            ar_impulse_response = lfilter([1], self.paramset_tail_modes_ar_coeff, impulse, axis=0)
            plotnorm(x=self.sampling_freq, y=ar_impulse_response)

    def late_tail_generation(self, fix_filter_nans=True, no_fade=False, doplots=False,
                             scaled_fade_in=False):
        """

            Generates the reverberant tai lfor the AIR being modelled. Involves a decaying tail,
            that fades in up to to the point of transition where it is at its maximum. It is
            consequently filtered by an AR filter to model the room modes

            Args:
            fix_filter_nans: Remove NaNs from the result
            no_fade: Do not apply a fading in, which is used to blend the early with the late part
            doplots: Flag to ask for plot to be drawn
            scaled_fade_in: Coefficient to scale the fade in point. Can be used to make the
            transition faster or slower

            Returns:
            The reverberant tail for the AIR being modelled
            The number of parameters needed for the representation

        """

        from scipy.signal import tf2zpk, zpk2tf

        nparams = 1  # t60
        tail_fame_overlap = 2.

        def tail_decay_func(x):
            return np.exp(
                -x / float(self.sampling_freq) * (3 * np.log(10) /
                                                  self.paramset_reverberation_time))

        def early_envelope_func(x):
            if no_fade:
                return tail_decay_func(x)
            if scaled_fade_in:
                time_scaler = (self.paramset_reverberation_time /
                               ((self.transition_sample - self.paramset_directsound_soa)
                                / float(self.sampling_freq)))
            else:
                time_scaler = 1.
            return tail_decay_func(
                (self.transition_sample - x) * time_scaler + self.transition_sample)

        # early_envelope_func = lambda x: tail_decay_func(x)

        def nan_check(x):
            if np.any(np.isnan(x)):
                raise AssertionError('Late tail has NaN values')

        self.printer(getfname() + " : Modelling reverberant tail")

        excitation, _ = self.excitation_model_reconstruct()
        excitation = row_vector(excitation)

        tail_nsamples = int(np.ceil(self.original.size))
        direct_arrive = self.paramset_directsound_soa
        transition_sample = int(np.ceil(self.transition_sample))

        tailsamples = np.zeros(tail_nsamples)
        tailsamples[int(transition_sample):] = tail_decay_func(
            np.arange(self.transition_sample, self.original.size, 1))
        early_envelope_samples = np.zeros((tail_nsamples,))
        early_env_samples = early_envelope_func(
            np.arange(int(np.round(direct_arrive)), transition_sample, 1))
        early_envelope_samples[int(np.round(direct_arrive)):transition_sample] = early_env_samples

        early_envelope_and_tail_1d = tailsamples + early_envelope_samples
        early_envelope_and_tail = np.array(early_envelope_and_tail_1d)

        nan_check(early_envelope_and_tail)
        final_tail_model = enframe(np.zeros((int(tail_fame_overlap * self.original.size),)),
                                   self.late_proc_framelength,
                                   np.ceil(self.late_proc_framelength))
        for i in range(final_tail_model.shape[0]):
            env_idx = i / float(final_tail_model.shape[0]) * early_envelope_and_tail.size
            these_scales = early_envelope_and_tail[
                           int(env_idx):int(env_idx) + final_tail_model.shape[1]]
            these_scales = np.atleast_2d(these_scales * np.random.normal(
                size=these_scales.size) * self.late_tail_gaussian_std).T
            these_delays = np.atleast_2d(
                np.sort(np.arange(-.5 * final_tail_model.shape[1],
                                  1.5 * final_tail_model.shape[1],
                                  2 * final_tail_model.shape[1] / float(
                                      these_scales.size)))).T
            these_delays_noise = np.random.normal(size=these_delays.size) / (
                    2 * final_tail_model.shape[1] / float(these_scales.size)) / 2.
            these_delays = these_delays + np.atleast_2d(these_delays_noise).T
            final_tail_model[i, :] = gm_frac_delayed_copies(
                these_scales[0:these_delays.size], these_delays[0:these_scales.size],
                self.late_proc_framelength,
                excitation_signal=excitation)[0]
        nan_check(final_tail_model)

        final_tail_model, _ = overlapadd(final_tail_model,
                                         np.ones(self.late_proc_framelength),
                                         np.ceil(
                                             self.late_proc_framelength / tail_fame_overlap))
        nan_check(final_tail_model)
        final_tail_model /= abs(final_tail_model).max()
        nan_check(final_tail_model)
        nparams += np.array(self.paramset_tail_modes_ar_coeff).size
        nparams += np.array(self.paramset_tail_modes_ma_coeff).size
        z, poles, k = tf2zpk(self.paramset_tail_modes_ma_coeff, self.paramset_tail_modes_ar_coeff)
        self.printer('Tail Poles radius: ' + float2str(abs(poles)))
        if np.any(abs(poles) >= 1):
            self.printer('Will remove unstable poles')
            poles = poles[abs(poles) < 1]
            b, a = zpk2tf(z, poles, k)
            self.paramset_tail_modes_ar_coeff = a / b[0]
            self.printer('New Tail Poles radius: ' + float2str(abs(poles)))
        # final_tail_model_no_filt = np.array(final_tail_model)
        final_tail_model = lfilter(self.paramset_tail_modes_ma_coeff,
                                   self.paramset_tail_modes_ar_coeff,
                                   final_tail_model, axis=0)
        if final_tail_model[np.abs(final_tail_model).argmax()] < 0:
            final_tail_model *= -1.
        ir = np.zeros(101)
        ir[50] = 1.
        ir = lfilter([1], self.paramset_tail_modes_ar_coeff,
                     ir, axis=0)
        shift = int(np.abs(ir).argmax() - 50.)
        if not shift == 0:
            self.printer('AR filter will be anti-causal because'
                         ' there is a shift of ' + str(shift))
            final_tail_model = np.concatenate((final_tail_model[shift:], np.zeros(shift)))
        elif shift < 0:
            raise AssertionError('Unexpected condition')
        else:
            self.printer('AR filter does not cause time distortion')

        ##
        if doplots:
            from utils_base import plotnorm
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 4))
            plotnorm(x=self.sampling_freq,
                     y=final_tail_model[:], title='', interactive=False, clf=True, )
            plotnorm(x=self.sampling_freq,
                     y=early_envelope_and_tail_1d, title='', interactive=False, clf=False, )
            plt.plot(direct_arrive / float(self.sampling_freq),
                     early_envelope_and_tail_1d[int(direct_arrive)] /
                     np.abs(early_envelope_and_tail_1d).max(), '+')
            plt.plot(transition_sample / float(self.sampling_freq),
                     0, '+')
            plt.legend(('Tail', 'Envelope', 'Direct sound TOA', 'Transition Time'))
            plt.tight_layout()
            fname = ('/tmp/fade_late_only.pdf'
                     if not no_fade else '/tmp/no_fade_late_only.pdf')
            plt.savefig(fname)
            print('Saved : ' + fname)
            plt.show()
            plt.close('all')

            #
            plt.figure(figsize=(6, 2))
            plotnorm(x=self.sampling_freq,
                     y=early_envelope_and_tail_1d, title='', interactive=False,
                     clf=True)
            plt.plot(direct_arrive / float(self.sampling_freq),
                     early_envelope_and_tail_1d[int(direct_arrive)] /
                     np.abs(early_envelope_and_tail_1d).max(), 'o', mfc='none')
            plt.plot(transition_sample / float(self.sampling_freq),
                     1., 'o', mfc='none')
            plt.legend(('Tail envelope', 'Direct sound TOA', 'Transition Time'))
            plt.tight_layout()
            fname = '/tmp/envelope.pdf'
            plt.savefig(fname)
            print('Saved : ' + fname)
            plt.show()
            plt.close('all')

        ##

        if fix_filter_nans:
            nan_points = np.where(np.isnan(final_tail_model))[-1]
            if nan_points.size > 0:
                print('Tail reco: Fixing to 0, ' + str(
                    final_tail_model.size - nan_points[0]) + ' points '
                      + 'out of ' + str(final_tail_model.size))
                final_tail_model[nan_points[0]:] = 0
        nan_check(final_tail_model)
        if final_tail_model.size < self.original.size:
            final_tail_model = np.concatenate((final_tail_model, np.zeros(final_tail_model.size -
                                                                          self.original.size)))
        elif final_tail_model.size > self.original.size:
            final_tail_model = final_tail_model[0:self.original.size]
        nan_check(final_tail_model)
        if np.any(np.isnan(final_tail_model)):
            raise AssertionError('Late tail has NaN values')

        return final_tail_model, nparams

    def air_reconstruct(self, assert_on_0_energy=True, fix_nans_in_scaling=True):
        """
        Reconstructs the entire AIR from the model.

        Args:
            assert_on_0_energy: Assert if the model does not contain any energy
            fix_nans_in_scaling: Fix and NaNs in the representation

        Returns:
            The AIR reconstruction
            The numebr of parameters in the model

        """
        self.printer(getfname() + " : Reconstructing AIR")
        air_early_part, air_direct_sound, nparams = self.early_part_reconstruct(
            desired_length=self.original.size if self.model_late_part else None,
            assert_on_0_energy=assert_on_0_energy, no_rescaling=True)

        if np.any(np.isnan(air_early_part)):
            raise AssertionError('Found NaN in early part reconstruction')

        def local_norm(x, tscale=None):
            if tscale is None:
                tscale = abs(x).max()
            if np.isnan(tscale):
                raise AssertionError('Scale is NaN')
            if tscale > 0:
                x = x / tscale
            return x

        if self.model_late_part:
            air_reflections = air_early_part - air_direct_sound
            air_reflections = local_norm(air_reflections, abs(air_early_part).max())
            air_direct_sound = local_norm(air_direct_sound, abs(air_direct_sound).max())
            self.printer(getfname() + " : Energy balancing early and late parts of AIR")
            direct_sound_energy = get_array_energy(air_direct_sound)
            early_ref_energy = get_array_energy(air_reflections)
            nparams += 1
            air_energy_scaling_early = np.sqrt(direct_sound_energy / (
                    self.paramset_drr_linscale_early * early_ref_energy))
            if air_energy_scaling_early < 0:
                raise NameError(getfname() + "ComplexEnergyScale")
            air_late_part, nparams_late = self.late_tail_generation(doplots=False)
            nparams += nparams_late
            if abs(air_late_part).max() == 0:
                raise AssertionError('The max tail sample is 0')
            air_late_part = local_norm(air_late_part, abs(air_late_part).max())
            if np.any(np.isnan(air_late_part)):
                if fix_nans_in_scaling:

                    nan_points = np.where(np.isnan(air_late_part))[-1]
                    if nan_points.size > 0:
                        print('Reco: Fixing to 0, ' + str(
                            air_late_part.size - nan_points[0]) + ' points '
                              + 'out of ' + str(air_late_part.size))
                        air_late_part[nan_points[0]:] = 0
                else:
                    raise AssertionError('The tail has nans')
            tail_energy = get_array_energy(air_late_part)
            if tail_energy == 0 and assert_on_0_energy:
                raise AssertionError('The energy of the tail is 0')
            nparams += 1
            air_energy_scaling_late = np.sqrt(direct_sound_energy / (
                    self.paramset_drr_linscale_late * tail_energy *
                    self.tail_visible_energy_ratio))
            if air_energy_scaling_late < 0:
                raise NameError(getfname() + "ComplexEnergyScale")
            if np.isnan(air_energy_scaling_late):
                raise AssertionError('The scaling coefficient for the tail is NaN')
            if np.isinf(air_energy_scaling_late):
                raise AssertionError('The scaling coefficient for the tail is inf')
            air_late_part *= air_energy_scaling_late
            ##
            # air_late_part[0:self.transition_sample] = 0
            # air_late_part = air_late_part / np.abs(air_late_part).max()
            ##
            if np.any(air_reflections > 0):
                air_reflections *= air_energy_scaling_early
        else:
            self.air_reconstructed_from_model = air_early_part
            self.number_of_parameters = nparams
            return np.array(air_early_part), nparams

        if np.any(np.isnan(air_direct_sound)):
            raise AssertionError('Found NaN in direct sound reconstruction')
        if not np.any(air_direct_sound > 0):
            self.printer('Found no non-zero values in direct sound reconstruction')
        if np.any(np.isnan(air_reflections)):
            raise AssertionError('Found NaN in early reflections reconstruction')
        if not np.any(air_reflections == 0):
            self.printer('Found NaN in early reflections reconstruction')
        if np.any(np.isnan(air_late_part)):
            raise AssertionError('Found NaN in late reflections reconstruction')

        self.air_reconstructed_from_model = air_direct_sound + air_reflections + air_late_part
        self.number_of_parameters = nparams

        return np.array(self.air_reconstructed_from_model), nparams

    def get_eval_scores(self, verbose=False, ref=None, rec=None, label=None, edc_comp=False,
                        ):
        """
        Returns a number of measures for the modeling quality when compared to the original AIR.

        Args:
            verbose: Verbose reporting
            ref: The reference AIR
            rec: The reconstruction
            label: The name of this model
            edc_comp: Flag for EDC compensation

        Returns:
            The score names
            The score values

        """

        if self.model_late_part:
            if ref is None:
                ref = np.array(self.original)
            if rec is None:
                rec = np.array(self.air_reconstructed_from_model)
        else:
            if ref is None:
                ref = self.original[0:self.transition_sample]
            if rec is None:
                rec = self.air_reconstructed_from_model[0:self.transition_sample]

        if edc_comp:
            edc_scale, ref = self.get_edc_scale(up_to=ref.size, reference=ref)
            rec = rec / edc_scale

        nparams = self.number_of_parameters
        fs = self.sampling_freq

        return get_eval_scores(ref, rec, fs, verbose=verbose, nparams=nparams, name=self.name,
                               label=label, rescale=True)

    def plot_modeling_results(self, saveloc=None, interactive=True, suptitle=None,
                              reconstruct=False, verbose=False):
        """
        Plots a summary of the fitting results for the AIR

        Args:
            saveloc: Location ot save the results in
            interactive: Wait for the user to close the plots
            suptitle: Title for the figure
            reconstruct: Rerun the reconstruction phase
            verbose: Verbose reporting

        Returns:
            Nothing

        """

        import matplotlib.pyplot as mplot
        from utils_spaudio import get_psd

        if reconstruct:
            self.air_reconstruct()
        else:
            try:
                if self.air_reconstructed_from_model.size == 0:
                    self.printer(
                        'You did not reconstruct the air before calling the plotter, so reconstructing')
                    self.air_reconstruct()
            except AttributeError:
                if self.air_reconstructed_from_model is None:
                    self.printer(
                        'You did not reconstruct the air before calling the plotter, so reconstructing')
                    self.air_reconstruct()

        self.print_reflection_report(verbose=verbose)

        fs = float(self.sampling_freq)
        t = np.arange(0, self.original.size) / fs
        y = self.air_reconstructed_from_model.flatten()
        mplot.close()
        mplot.figure(figsize=(6, 4))
        mplot.subplot('211')
        mplot.plot(t, np.array(self.original), label='AIR', linewidth=0.5)
        mplot.plot(t[0:y.size], y, label='Reconstructed',
                   linewidth=0.5)
        mplot.plot(self.paramset_directsound_soa / fs, 0, 'r+', markersize=2,
                   label='Est. Dir. Sound')
        mplot.plot(self.paramset_reflections_soa / fs, np.zeros(
            self.paramset_reflections_soa.size), 'k+', markersize=2,
                   label='Est. Reflections(' + str(self.paramset_reflections_soa.size) + ')')
        mplot.xlabel('Time (s)')

        dsoa = (self.paramset_directsound_soa / float(self.sampling_freq))

        lowx = .6 * dsoa
        if dsoa <= 0.003:
            lowx = -0.002
        if self.model_late_part:
            mplot.xlim([lowx,
                        self.paramset_reverberation_time / 3.
                        + self.defined_transition_time * 1.1 + dsoa])
        else:
            mplot.xlim([lowx,
                        (self.paramset_directsound_soa / float(self.sampling_freq)) +
                        self.defined_transition_time * 1.1])

        mplot.legend(loc='upper right', ncol=2, fontsize=8)
        mplot.grid()
        mplot.subplot('212')
        if self.model_late_part:
            ref = np.array(self.original)
            rec = self.air_reconstructed_from_model
        else:
            ref = self.original[0:self.transition_sample]
            rec = self.air_reconstructed_from_model[0:self.transition_sample]

        ref = ref / abs(ref).max()
        rec = rec / abs(rec).max()
        f, ref_psd, _ = get_psd(ref, self.sampling_freq, 0.002)
        _, rec_psd, _ = get_psd(rec, self.sampling_freq, 0.002)
        scale = max(ref_psd.max(), rec_psd.max())
        ref_psd_plt = 10. * np.log10(ref_psd / scale)
        rec_psd_plt = 10. * np.log10(rec_psd / scale)
        scale = np.linalg.lstsq(np.atleast_2d(ref_psd_plt).T,
                                np.atleast_2d(rec_psd_plt).T, rcond=None)[0][0]
        ref_psd_plt = ref_psd_plt * scale
        mplot.semilogx(f, ref_psd_plt, '-', label='Original PSD', linewidth=0.5, )
        mplot.semilogx(f, rec_psd_plt, '-', label='Reconstruction PSD', linewidth=0.5)
        mplot.grid(True)
        mplot.xlabel('Frequency (Hz)')
        mplot.ylabel('V^2/Hz (dB)')
        mplot.legend(loc='lower left', ncol=2, fontsize=8)
        if suptitle is not None:
            mplot.suptitle(suptitle)
        mplot.tight_layout()

        if saveloc is not None:
            mplot.savefig(saveloc)
            self.printer('Saved: ' + saveloc, force_loud=verbose)
        if interactive:
            mplot.show()

    def print_reflection_report(self, verbose=False):
        """
        Prints a report of the reflections modeled

        Args:
            verbose: Verbose reporting

        Returns:
            Nothing

        """
        from tabulate import tabulate

        if self.model_late_part:
            self.printer('AR coeff: ' + float2str(self.paramset_tail_modes_ar_coeff, 15))
            self.printer('MA coeff: ' + float2str(self.paramset_tail_modes_ma_coeff, 15))
        print('Excitation : ' + float2str([0, 1, 0] if self.use_delta_as_excitation else
                                          self.excitation_model_reconstruct()[0].flatten(), 15))

        self.printer('Direct Sound', force_loud=verbose)
        dir_toa = self.paramset_directsound_soa / float(self.sampling_freq)
        ds_tab = tabulate(np.atleast_2d(np.concatenate((
            np.atleast_1d(dir_toa),
            np.atleast_1d(self.paramset_directsound_soa),
            np.atleast_1d(self.paramset_directsound_amp)
        ))),
            headers=('ToA', 'SOA', 'Amplitude'), showindex=True)
        self.printer(ds_tab, force_loud=verbose)
        ref_tab = tabulate(np.concatenate((np.atleast_2d(self.paramset_reflections_soa /
                                                         float(self.sampling_freq)).T,
                                           np.atleast_2d(self.paramset_reflections_soa).T,
                                           np.atleast_2d(self.paramset_reflections_soa /
                                                         float(self.sampling_freq)).T - dir_toa,
                                           np.atleast_2d(self.paramset_reflections_amp).T,
                                           np.atleast_2d(
                                               self.paramset_reflections_fval_reduction).T,
                                           ), axis=1),
                           headers=['ToA', 'SOA', 'TOA (r)', 'Amplitude', 'dFval'],
                           showindex=True)
        self.printer('Reflections', force_loud=verbose)
        self.printer(ref_tab, force_loud=verbose)

    def baseline_reports(self, nparams_required, do_plots=False, interactive=False,
                         savefig_loc=None, cap_at_final=True):
        """
        Models the AIR using a set of baselines and compares the accuracy of modeling the
        environment using those baselines and this model

        Args:
            nparams_required: The number of parameters to use for each baseline model
            do_plots: Flag for plotting
            interactive: Wait for user to close plots
            savefig_loc: Location to save the figures in
            cap_at_final: Cap the number of parameters used by each model to the number of
            parameters of this model

        Returns:
            Nothing

        """

        if cap_at_final:
            nparams_final = self.air_reconstruct()[1]
            if nparams_required > nparams_final:
                nparams_required = nparams_final
                self.printer('Adjusted number of parameters for baselines to the lower limit of ' +
                             str(nparams_required) + ' to be the same as the final model')

        model_reco = self.air_reconstruct()[0]
        if self.model_late_part:
            ref = np.array(self.original)
        else:
            ref = self.original[0:self.transition_sample]
        model_reco = model_reco[0:ref.size]
        self.get_eval_scores(verbose=True, ref=ref, rec=model_reco, label='Final')

        ref = ref / abs(ref).max()

        # Sparse
        strong_n = (-abs(ref)).argsort()[0:int(np.ceil(nparams_required / 2.))]
        reco_sparse = np.zeros_like(ref)
        reco_sparse[strong_n] = ref[strong_n]
        _, scores_sp = self.get_eval_scores(verbose=True, ref=ref, rec=reco_sparse, label='Sparse')
        # Truncated FIR
        trunc_n = (np.round(self.paramset_directsound_soa) + np.array(
            range(nparams_required - 1))).astype(int)
        reco_trunc = np.zeros_like(ref)
        trunc_n = trunc_n[trunc_n < ref.size]
        reco_trunc[trunc_n] = ref[trunc_n]
        _, scores_trunc = self.get_eval_scores(verbose=True, ref=ref, rec=reco_trunc,
                                               label='Truncation')

        if do_plots:

            from utils_base import plotnorm, float2str
            import matplotlib.pyplot as plt

            plt.figure(figsize=(19.2 * .3, 9.43 * .3))
            plt.subplot('211')
            plotnorm(x=self.sampling_freq, y=ref, label='Original', title='')
            plotnorm(x=self.sampling_freq, y=model_reco,
                     label='Model',  # Num. Param.' + str(nparams_required),
                     title='')
            plt.ylabel('')
            plt.xlabel('')
            self.set_plot_bounds()
            plt.legend(loc='upper right', fontsize=8)

            plt.subplot('212')
            plotnorm(x=self.sampling_freq, y=ref, label='Original', title='')
            plotnorm(x=strong_n / float(self.sampling_freq), y=reco_sparse[strong_n],
                     label='Sparse', marker='.', linestyle='', markersize=2,
                     title='')
            self.set_plot_bounds()
            plt.ylabel('')
            plt.xlabel('')
            plt.xlabel('Time (s)')
            plt.legend(loc='upper right', fontsize=8)

            plt.tight_layout()

            if savefig_loc is not None:
                plt.savefig(savefig_loc)
                print('Saved: ' + savefig_loc)
            if interactive:
                plt.show()

    def set_plot_bounds(self):
        """
        Deduces the time axis limits for plots

        Returns:
            Nothing

        """
        import matplotlib.pyplot as plt

        dsoa = (self.paramset_directsound_soa / float(self.sampling_freq))
        lowx = .6 * dsoa
        if dsoa <= 0.003:
            lowx = -0.002
        if self.model_late_part:
            plt.xlim([lowx,
                      self.paramset_reverberation_time / 3.
                      + self.defined_transition_time * 1.1 + dsoa])
        else:
            plt.xlim([lowx,
                      (self.paramset_directsound_soa / float(self.sampling_freq)) +
                      self.defined_transition_time * 1.1])


def plot_matched_excitation(or_dpath_in, excitation, interactive=False, fs=None, savelocation=None):
    import matplotlib.pyplot as plt
    from utils_spaudio import fractional_alignment, align_max_samples

    or_dpath = np.array(or_dpath_in)
    excitation_plt = np.array(excitation)
    aligned_mat = align_max_samples([or_dpath, excitation_plt])[0]
    center_sample = max([np.argmax(abs(aligned_mat[i, :])) for i in range(aligned_mat.shape[0])])
    back_samples_start = int(max(0, center_sample - np.ceil(excitation_plt.size / 2.)))
    front_samples_end = int(
        min(aligned_mat.shape[1], center_sample + np.ceil(excitation_plt.size / 2.)))
    or_dpath = aligned_mat[0, back_samples_start:front_samples_end]
    excitation_plt = aligned_mat[1, back_samples_start:front_samples_end]
    aligned_dpath, _, fscales = fractional_alignment(
        np.concatenate((np.atleast_2d(or_dpath), np.atleast_2d(excitation_plt)), axis=0),
        ls_scale=True, resolution=0.0001)

    plotnorm(x=fs, y=aligned_dpath[0], label='Original', interactive=False,
             no_rescaling=True)
    plotnorm(x=fs, y=fscales[1] * aligned_dpath[1, :],
             title='Excitation Model',
             no_rescaling=True,
             savelocation=savelocation,
             interactive=False, label='Excitation')
    plt.legend(fontsize=8)
    if interactive:
        plt.show()


def get_eval_scores(ref, rec, fs, verbose=False, nparams=None,
                    name=None, label=None, rescale=True, plot=False, savefig=None,
                    interactive=False):
    """
    Evaluates the modeling quality of an acoustic channel, when compared to the original AIR FIR
    filter taps.

    Args:
        ref: The original AIR
        rec: The reconstruction of the AIR
        fs: The sampling frequency
        verbose: Verbose reporting
        nparams: The number of parameters needed for the model used for hte reconstruction
        name: The name of the reconstruction
        label: The name of the model
        rescale: Normalize scales
        plot: Do plots
        savefig: Save figure here
        interactive: Wait for user to close the plot

    Returns:
        The score names
        The score values

    """
    from utils_spaudio import distitpf, get_psd
    from utils_reverb import npm

    score_names = ('DimensionalityReduction', 'NormalizedError', 'Correlation',
                   'ItakuraDistance', 'NPM')
    scores = [0., 0., 0., 0., 0.]

    if rescale:
        ref = ref * \
              np.linalg.lstsq(np.atleast_2d(ref).T, np.atleast_2d(rec).T, rcond=None)[0][0][0]

    ref_ps = get_psd(ref, fs, 0.005)[2]
    rec_ps = get_psd(rec, fs, 0.005)[2]
    if rescale:
        ref_ps = ref_ps * \
                 np.linalg.lstsq(np.atleast_2d(ref_ps).T,
                                 np.atleast_2d(rec_ps).T, rcond=None)[0][0][0]

    if nparams is None:
        nparams = np.inf
    nsamples = ref.size
    scores[0] = 1. - float(nparams) / nsamples
    scores[1] = np.sum(np.square(ref - rec)) / \
                np.sum(np.square(ref))
    scores[2] = np.corrcoef(ref, rec)[0, 1]
    scores[3] = distitpf(ref_ps, rec_ps)
    scores[4] = npm(ref_ps, rec_ps)
    if verbose:
        if name is None:
            name = 'AIR'
        if label is None:
            label = 'Experiment'
        print('For eval of ' + name)
        suffix = ''
        if label is not None:
            suffix = ' ' + label
        if verbose:
            print('Evaluation Measures' + suffix)
            for i in range(len(score_names)):
                print(score_names[i] + ', ' +
                      float2str(scores[i]) + ', ' +
                      float2str(10 * np.log10(scores[i])) + 'dB')

            print('Evaluation Measures Done')
    if plot:
        plotnorm(x=fs, y=ref, interactive=False, clf=True)

        plotnorm(x=fs, y=rec,
                 title='Referrence and Reconstruction NPM ' +
                       float2str(
                           10 * np.log10(scores[score_names.index('NPM')])) + 'dB',
                 interactive=interactive, savelocation=savefig)

    return score_names, tuple(scores)


def get_edc_scale(reference, up_to=None, edc_scale_lim=None):
    """
            Get the scaling vector to use for the compensation of the energy decay based on the EDC

            Args:
                up_to: Level up to which to get down to
                reference: The AIR to use for the scale estimation.

            Returns:
                The scale
    """

    go_back_to_1d = False
    if reference.ndim == 1:
        go_back_to_1d = True
        reference = np.atleast_2d(reference)
    this_edc_scale = np.zeros_like(reference)
    original_edc_scaled = np.zeros_like(reference)
    if up_to is None:
        up_to = reference.shape[1]
    reference = reference[:, 0:up_to]
    this_edc_scale = this_edc_scale[:, 0:up_to]
    original_edc_scaled = original_edc_scaled[:, 0:up_to]
    for i in range(reference.shape[0]):
        reference[i, :] = reference[i, :] / abs(reference[i, :]).max()
        edc_scale = pow(get_edc(np.array(reference[i, :])), 0.5)
        if not edc_scale.size == reference.shape[1]:
            raise AssertionError('Unexpected condition')
        if edc_scale_lim is not None:
            edc_scale[np.where(edc_scale < edc_scale_lim)] = edc_scale_lim
        original_edc_scaled[i, :] = reference[i, :] / edc_scale
        this_edc_scale[i, :] = edc_scale[0:up_to]
    if go_back_to_1d:
        this_edc_scale = this_edc_scale.flatten()
        original_edc_scaled = original_edc_scaled.flatten()

    return this_edc_scale, original_edc_scaled


def get_pca_excitation(airs, fs, modeling_span=(0.003, 0.003),
                       npccomps=.95, y=None, window=False,
                       auto_reduce=False, take_base_as=0):
    """
    Estimates the excitation used ot measure a set of AIRs, which were measured using the same
    measuring method and the same equipment. The approach is discussed in:
    C. Papayiannis, C. Evers, and P. A. Naylor, "Data Augmentation of Room Classifiers using Generative Adversarial Networks," arXiv preprint arXiv:1901.03257, 2018.

    Args:
        airs: The AIR FIR filter taps
        fs: The sampling frequency
        modeling_span: The length of the window of the excitation in seconds. This is indicated
        as a tuple, which shows the time span from the center of the excitation to the left and
        the right of the center
        npccomps: The number of principle components to use, or the proportion of the explained
        variance to ask from the principle components
        y: Provide this, if you want to model an AIR y using this method, otherwise the entire
        matrix of provided responses will be used
        window: The window samples
        auto_reduce: Automatically drop principle components if the explained variance is 100%
        take_base_as: Take this AIR index as the base for the alignments

    Returns:
        The estimated excitation(s)
        The aligned AIRs

    """
    from utils_spaudio import align_max_samples, fractional_alignment
    from utils_base import float2str
    from sklearn.decomposition import PCA

    airs = np.array(airs)
    dsample = 0
    for i in range(airs.shape[0]):
        dsample = max(dsample, abs(airs[i, :]).argmax())
    sample_span_front1 = int(np.ceil(fs * modeling_span[1]))
    front_span = min(dsample + sample_span_front1, airs.shape[1])

    airs_max_aligned = align_max_samples(airs[:, 0:front_span])[0]

    dsample = 0
    for i in range(airs_max_aligned.shape[0]):
        dsample = max(dsample, abs(airs_max_aligned[0, :]).argmax())
    sample_span_back = int(np.ceil(fs * modeling_span[0]))
    sample_span_front2 = int(np.ceil(fs * modeling_span[1]))
    front_span = min(dsample + sample_span_front2, airs_max_aligned.shape[1])
    back_span = max(dsample - sample_span_back, 0)

    airs_or_out = np.array(airs_max_aligned[:, back_span:front_span])

    if airs_or_out.shape[0] == 1:
        print('Cannot do PCA based modelling as you only gave me one AIR')
        return airs_or_out, airs_or_out

    airs = airs_max_aligned[:, back_span:front_span]
    airs, fdelays, _ = fractional_alignment(airs, ls_scale=True, take_base_as=take_base_as)
    print('Fractional alignment delays: ' + float2str(fdelays))

    if npccomps < 1:
        ncomponentes = None
        print('Will find number of components that explains more than ' +
              str(npccomps) + ' of the variance')
    else:
        ncomponentes = npccomps
    pca = PCA(n_components=ncomponentes, svd_solver='full', copy=True)
    pca.fit(np.array(airs))

    if npccomps < 1:
        if npccomps is not None:
            cum_exp_var = np.cumsum(pca.explained_variance_ratio_)
            print('Explained Variance (cumulative): ' + float2str(cum_exp_var))
            npccomps = np.where(cum_exp_var > npccomps)[-1][0]
            var_explained = cum_exp_var[npccomps]
            if npccomps > 0 and var_explained == 1. and auto_reduce:
                npccomps -= 1
                print('Reduced PCA components to ' + str(
                    npccomps) + ' to avoid capturing 100% of the '
                                'variance which is undesirable')
    var_explained = np.cumsum(pca.explained_variance_ratio_)[npccomps]
    print('New explained variance is ' + float2str(var_explained))

    if y is None:
        trans_air = pca.transform(np.array(airs))
    else:
        trans_air = pca.transform(y)
    trans_air[:, npccomps:] = 0
    trans_air = pca.inverse_transform(trans_air)

    print('PCA chosen ' + str(npccomps + 1) + ' components which explains ' + float2str(
        var_explained) + ' of the variance')

    if window:
        window_samples = np.hamming(trans_air.shape[1])
        for i in range(trans_air.shape[0]):
            trans_air[i, :] *= window_samples

    return trans_air, airs_or_out
