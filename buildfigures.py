import figures.mplhelpers

import figures.test.plot as plot_test
import figures.mean_field.plot as plot_mean_field
import figures.bec_noise.plot as plot_bec_noise
import figures.squeezing.plot as plot_squeezing
import figures.exact.plot as plot_exact
import figures.bell.plot as plot_bell

FOLDER = 'figures_generated/'
FORMAT = '.pdf'


if __name__ == '__main__':

    # Testing plots
    for wigner in (False, True):
        for abbrev in (
            "CDIP", "RK4IP", "RK46NL", "CD"
            ):
            plot_test.convergence(wigner, abbrev,
                FOLDER + 'test/convergence_' +
                ('wigner_' if wigner else 'gpe_') +
                abbrev + FORMAT)
    plot_test.convergence_by_time(FOLDER + 'test/convergence_by_time' + FORMAT)
    plot_test.grid_check_gpe(FOLDER + 'test/grid_check_gpe' + FORMAT)
    plot_test.grid_check_wigner(FOLDER + 'test/grid_check_wigner' + FORMAT)

	# Mean-field ground states
    plot_mean_field.one_comp_gs_small(FOLDER + 'mean_field/one_comp_gs_small' + FORMAT)
    plot_mean_field.one_comp_gs_large(FOLDER + 'mean_field/one_comp_gs_large' + FORMAT)
    plot_mean_field.two_comp_gs_immiscible(FOLDER + 'mean_field/two_comp_gs_immiscible' + FORMAT)
    plot_mean_field.two_comp_gs_miscible(FOLDER + 'mean_field/two_comp_gs_miscible' + FORMAT)

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
    plot_squeezing.feshbach_squeezing_no_losses(FOLDER + 'bec_squeezing/feshbach_squeezing_no_losses' + FORMAT)

    # Exact comparison
    plot_exact.squeezing_nocc_100(FOLDER + 'exact/squeezing_nocc_100' + FORMAT)
    plot_exact.squeezing_cc_100(FOLDER + 'exact/squeezing_cc_100' + FORMAT)
    plot_exact.squeezing_nocc_err(FOLDER + 'exact/squeezing_nocc_err' + FORMAT)
    plot_exact.squeezing_cc_err(FOLDER + 'exact/squeezing_cc_err' + FORMAT)
    plot_exact.squeezing_nocc_N_err(FOLDER + 'exact/squeezing_nocc_N_err' + FORMAT)
    plot_exact.squeezing_cc_N_err(FOLDER + 'exact/squeezing_cc_N_err' + FORMAT)

    # Bell inequalities
    plot_bell.cooperative_N1(FOLDER + 'bell/cooperative_N1' + FORMAT)
    plot_bell.cooperative_N2(FOLDER + 'bell/cooperative_N2' + FORMAT)

    plot_bell.distribution_Q1(FOLDER + 'bell/distribution_Q1' + FORMAT)
    plot_bell.distribution_Q2(FOLDER + 'bell/distribution_Q2' + FORMAT)
    plot_bell.distribution_P1(FOLDER + 'bell/distribution_P1' + FORMAT)
    plot_bell.distribution_P2(FOLDER + 'bell/distribution_P2' + FORMAT)

    plot_bell.ghz_violations(FOLDER + 'bell/ghz_violations' + FORMAT)
    plot_bell.ghz_errors(FOLDER + 'bell/ghz_errors' + FORMAT)

