import figures.mplhelpers

import figures.mean_field.plot as plot_mean_field
import figures.phase_noise.plot as plot_phase_noise

FOLDER = 'figures_generated/'
FORMAT = '.pdf'


if __name__ == '__main__':

	# Mean-field ground states
    plot_mean_field.one_comp_gs_small(FOLDER + 'mean_field/one_comp_gs_small' + FORMAT)
    plot_mean_field.one_comp_gs_large(FOLDER + 'mean_field/one_comp_gs_large' + FORMAT)
    plot_mean_field.two_comp_gs_immiscible(FOLDER + 'mean_field/two_comp_gs_immiscible' + FORMAT)
    plot_mean_field.two_comp_gs_miscible(FOLDER + 'mean_field/two_comp_gs_miscible' + FORMAT)

    # Visibility and phase noise
    plot_phase_noise.ramsey_short(FOLDER + 'phase_noise/ramsey_visibility_short' + FORMAT)
    plot_phase_noise.spinecho_short(FOLDER + 'phase_noise/echo_visibility_short' + FORMAT)
    plot_phase_noise.ramsey_long(FOLDER + 'phase_noise/ramsey_visibility_long' + FORMAT)
    plot_phase_noise.spinecho_long(FOLDER + 'phase_noise/echo_visibility_long' + FORMAT)

    plot_phase_noise.ramsey_noise(FOLDER + 'phase_noise/ramsey_noise' + FORMAT)
    plot_phase_noise.spinecho_noise(FOLDER + 'phase_noise/echo_noise' + FORMAT)
