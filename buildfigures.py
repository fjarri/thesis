import figures.mplhelpers

import figures.mean_field.plot as plot_mean_field
import figures.bec_noise.plot as plot_bec_noise
import figures.squeezing.plot as plot_squeezing

FOLDER = 'figures_generated/'
FORMAT = '.pdf'


if __name__ == '__main__':

	# Mean-field ground states
    """
    plot_mean_field.one_comp_gs_small(FOLDER + 'mean_field/one_comp_gs_small' + FORMAT)
    plot_mean_field.one_comp_gs_large(FOLDER + 'mean_field/one_comp_gs_large' + FORMAT)
    plot_mean_field.two_comp_gs_immiscible(FOLDER + 'mean_field/two_comp_gs_immiscible' + FORMAT)
    plot_mean_field.two_comp_gs_miscible(FOLDER + 'mean_field/two_comp_gs_miscible' + FORMAT)
    """

    # Visibility and phase noise
    plot_bec_noise.ramsey_short(FOLDER + 'bec_noise/ramsey_visibility_short' + FORMAT)
    plot_bec_noise.spinecho_short(FOLDER + 'bec_noise/echo_visibility_short' + FORMAT)
    plot_bec_noise.ramsey_long(FOLDER + 'bec_noise/ramsey_visibility_long' + FORMAT)
    plot_bec_noise.spinecho_long(FOLDER + 'bec_noise/echo_visibility_long' + FORMAT)

    plot_bec_noise.ramsey_single_run_population(FOLDER + 'bec_noise/ramsey_single_run_pop' + FORMAT)
    plot_bec_noise.spinecho_single_run_population(FOLDER + 'bec_noise/echo_single_run_pop' + FORMAT)

    plot_bec_noise.ramsey_noise(FOLDER + 'bec_noise/ramsey_noise' + FORMAT)
    plot_bec_noise.spinecho_noise(FOLDER + 'bec_noise/echo_noise' + FORMAT)
    plot_bec_noise.illustration_noise(FOLDER + 'bec_noise/illustration_noise_20ms' + FORMAT, 20)
    plot_bec_noise.illustration_noise(FOLDER + 'bec_noise/illustration_noise_450ms' + FORMAT, 450)

    # Squeezing
    plot_squeezing.riedel_rotation(FOLDER + 'bec_squeezing/riedel_rotation' + FORMAT)
    plot_squeezing.riedel_cloud(FOLDER + 'bec_squeezing/riedel_cloud' + FORMAT)
    plot_squeezing.feshbach_scattering(FOLDER + 'bec_squeezing/feshbach_scattering' + FORMAT)
    plot_squeezing.feshbach_squeezing(FOLDER + 'bec_squeezing/feshbach_squeezing' + FORMAT)
