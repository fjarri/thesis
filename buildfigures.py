import figures.mplhelpers

import figures.mean_field.plot as plot_mean_field

FOLDER = 'figures_generated/'
FORMAT = '.pdf'


if __name__ == '__main__':

	# Mean-field ground states
    plot_mean_field.one_comp_gs_small(FOLDER + 'mean_field/one_comp_gs_small' + FORMAT)
    plot_mean_field.one_comp_gs_large(FOLDER + 'mean_field/one_comp_gs_large' + FORMAT)
    plot_mean_field.two_comp_gs_immiscible(FOLDER + 'mean_field/two_comp_gs_immiscible' + FORMAT)
    plot_mean_field.two_comp_gs_miscible(FOLDER + 'mean_field/two_comp_gs_miscible' + FORMAT)
