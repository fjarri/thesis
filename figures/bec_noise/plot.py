import numpy
import json
from scipy.interpolate import interp1d

from figures import get_path
import figures.mplhelpers as mplh


def ramsey_short(fname):
    with open(get_path(__file__, 'visibility/ramsey_experimental.json')) as f:
        vis_exp = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_icols2011.json')) as f:
        icols2011 = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_gpe_vis.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_wigner_vis.json')) as f:
        wig = json.load(f)

    t_exp = numpy.array(vis_exp['xarray'])
    v_exp = numpy.array(vis_exp['yarray'])
    v_exp_err = numpy.array(vis_exp['yerrors'])

    all_noise_t = numpy.array(icols2011['times'])
    all_noise_v = numpy.array(icols2011['vis_classical_noise'])

    gpe_t = numpy.array(gpe['times'])
    gpe_v = numpy.array(gpe['visibility'])
    wig_t = numpy.array(wig['times'])
    wig_v = numpy.array(wig['visibility'])
    wig_v_err = numpy.array(wig['visibility_errors'])

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$T$ (s)',
        ylabel='$\\mathcal{V}$')
    s.errorbar(t_exp, v_exp, yerr=v_exp_err, color='k', linestyle='none',
        capsize=1.5)

    # GPE
    s.plot(gpe_t, gpe_v, color=mplh.color.f.red.main,
        linestyle=':', dashes=mplh.dash[':'])
    # Pure Wigner
    s.plot(wig_t, wig_v, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])
    # Wigner + technical noise
    s.plot(all_noise_t, all_noise_v, color=mplh.color.f.green.main,
        linestyle='--', dashes=mplh.dash['--'])

    s.set_xlim((0, 1.3))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def spinecho_short(fname):
    with open(get_path(__file__, 'visibility/echo_experimental.json')) as f:
        vis_exp = json.load(f)
    with open(get_path(__file__, 'visibility/echo_icols2011.json')) as f:
        icols2011 = json.load(f)
    with open(get_path(__file__, 'visibility/echo_gpe_vis.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/echo_wigner_vis.json')) as f:
        wig = json.load(f)

    t_exp = numpy.array(vis_exp['xarray'])
    v_exp = numpy.array(vis_exp['yarray'])
    v_exp_err = numpy.array(vis_exp['yerrors'])

    all_noise_t = numpy.array(icols2011['times'])
    all_noise_v = numpy.array(icols2011['vis_classical_noise'])

    gpe_t = numpy.array(gpe['times'])
    gpe_v = numpy.array(gpe['visibility'])
    wig_t = numpy.array(wig['times'])
    wig_v = numpy.array(wig['visibility'])
    wig_v_err = numpy.array(wig['visibility_errors'])

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$T$ (s)',
        ylabel='$\\mathcal{V}$')
    s.errorbar(t_exp, v_exp, yerr=v_exp_err, color='k', linestyle='none',
        capsize=1.5)

    # GPE
    s.plot(gpe_t, gpe_v, color=mplh.color.f.red.main,
        linestyle=':', dashes=mplh.dash[':'])
    # Pure Wigner
    s.plot(wig_t, wig_v, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])
    # Wigner + technical noise
    s.plot(all_noise_t, all_noise_v, color=mplh.color.f.green.main,
        linestyle='--', dashes=mplh.dash['--'])

    s.set_xlim((0, 1.8))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def ramsey_long(fname):
    with open(get_path(__file__, 'visibility/ramsey_long_gpe_vis.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_long_wigner_vis.json')) as f:
        wig = json.load(f)

    t_gpe = numpy.array(gpe['xarray'])
    v_gpe = numpy.array(gpe['yarray'])
    t_wig = numpy.array(wig['xarray'])
    v_wig = numpy.array(wig['yarray'])

    fig = mplh.figure()
    s = fig.add_subplot(111,
        xlabel='$T$ (s)',
        ylabel='$\\mathcal{V}$')

    # GPE
    s.plot(t_gpe, v_gpe, color=mplh.color.f.red.main,
        linestyle=':', dashes=mplh.dash[':'])
    # Pure Wigner
    s.plot(t_wig, v_wig, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 5.))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def spinecho_long(fname):
    with open(get_path(__file__, 'visibility/echo_long_gpe_vis.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/echo_long_wigner_vis.json')) as f:
        wig = json.load(f)

    t_gpe = numpy.array(gpe['xarray'])
    v_gpe = numpy.array(gpe['yarray'])
    t_wig = numpy.array(wig['xarray'])
    v_wig = numpy.array(wig['yarray'])

    fig = mplh.figure()
    s = fig.add_subplot(111,
        xlabel='$T$ (s)',
        ylabel='$\\mathcal{V}$')

    # GPE
    s.plot(t_gpe, v_gpe, color=mplh.color.f.red.main,
        linestyle=':', dashes=mplh.dash[':'])
    # Pure Wigner
    s.plot(t_wig, v_wig, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 5.))
    s.set_ylim((0, 1.))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def combine_noises(x_final, *pairs):
    interps = [
        interp1d(x, y, kind="cubic", bounds_error=False)(x_final) for x, y in pairs]

    res = interps[0] ** 2
    for i in interps[1:]:
        res += i ** 2

    return numpy.sqrt(res)


def ramsey_noise(fname):
    with open(get_path(__file__, 'noise/ramsey_phnoise_exp.json')) as f:
        exp = json.load(f)
    with open(get_path(__file__, 'noise/ramsey_imaging_noise_exp.json')) as f:
        imaging_exp = json.load(f)
    with open(get_path(__file__, 'noise/ramsey_wigner_noise.json')) as f:
        wig = json.load(f)
    with open(get_path(__file__, 'noise/ramsey_wigner_varied_pulse_noise.json')) as f:
        pulse_wig = json.load(f)

    t_wig = numpy.array(wig['times'])
    ph_wig = numpy.array(wig['phnoise'])

    t_pulse_wig = numpy.array(pulse_wig['times'])
    ph_pulse_wig = numpy.array(pulse_wig['phnoise'])

    t_exp = numpy.array(exp['xarray'])
    ph_exp = numpy.array(exp['yarray'])
    ph_exp_errors = numpy.array(exp['yerrors'])

    t_img_exp = numpy.array(imaging_exp['xarray'])
    ph_img_exp = numpy.array(imaging_exp['yarray'])

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$T$ (s)',
        ylabel='$\\mathcal{\\sigma}$ (rad)')

    s.errorbar(t_exp, ph_exp, yerr=ph_exp_errors, color='k', linestyle='none',
        capsize=1.5)

    s.plot(t_wig, ph_wig, color=mplh.color.f.blue.main,
        linestyle='--', dashes=mplh.dash['--'])
    s.plot(t_pulse_wig, ph_pulse_wig, color=mplh.color.f.blue.main,
        linestyle='-', dashes=mplh.dash['-'])

    s.plot(t_img_exp, ph_img_exp, color=mplh.color.f.red.main,
        linestyle=':', dashes=mplh.dash[':'])
    s.plot(t_img_exp, t_img_exp * 0.5, color=mplh.color.f.green.main,
        linestyle='-.', dashes=mplh.dash['-.'])
    s.plot(
        t_img_exp,
        combine_noises(t_img_exp,
            (t_img_exp, ph_img_exp),
            (t_pulse_wig, ph_pulse_wig),
            (t_img_exp, 0.5 * t_img_exp)), # see Egorov2011; MW freq instability of 0.5 rad/s
        color='k',
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 0.9))
    s.set_ylim((0, 0.55))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)


def spinecho_noise(fname):
    with open(get_path(__file__, 'noise/echo_phnoise_exp.json')) as f:
        exp = json.load(f)
    with open(get_path(__file__, 'noise/echo_imaging_noise_exp.json')) as f:
        imaging_exp = json.load(f)
    with open(get_path(__file__, 'noise/echo_wigner_noise.json')) as f:
        wig = json.load(f)
    #with open(get_path(__file__, 'noise/echo_wigner_varied_pulse_noise.json')) as f:
    #    pulse_wig = json.load(f)

    t_wig = numpy.array(wig['times'])
    ph_wig = numpy.array(wig['phnoise'])

    #t_pulse_wig = numpy.array(pulse_wig['times'])
    #ph_pulse_wig = numpy.array(pulse_wig['phnoise'])

    t_exp = numpy.array(exp['xarray'])
    ph_exp = numpy.array(exp['yarray'])
    ph_exp_errors = numpy.array(exp['yerrors'])

    t_img_exp = numpy.array(imaging_exp['xarray'])
    ph_img_exp = numpy.array(imaging_exp['yarray'])

    fig = mplh.figure(width=0.75)
    s = fig.add_subplot(111,
        xlabel='$T$ (s)',
        ylabel='$\\mathcal{\\sigma}$ (rad)')

    s.errorbar(t_exp, ph_exp, yerr=ph_exp_errors, color='k', linestyle='none',
        capsize=1.5)

    s.plot(t_wig, ph_wig, color=mplh.color.f.blue.main,
        linestyle='--', dashes=mplh.dash['--'])
    #s.plot(t_pulse_wig, ph_pulse_wig, color=mplh.color.f.blue.main,
    #    linestyle='-', dashes=mplh.dash['-'])

    s.plot(t_img_exp, ph_img_exp, color=mplh.color.f.red.main,
        linestyle=':', dashes=mplh.dash[':'])
    s.plot(t_img_exp, t_img_exp * 0.125, color=mplh.color.f.green.main,
        linestyle='-.', dashes=mplh.dash['-.'])
    s.plot(
        t_img_exp,
        combine_noises(t_img_exp,
            (t_img_exp, ph_img_exp),
            (t_wig, ph_wig),
            (t_img_exp, 0.125 * t_img_exp)), # see Egorov2011; MW freq instability of 0.125 rad/s
        color='k',
        linestyle='-', dashes=mplh.dash['-'])

    s.set_xlim((0, 1.6))
    s.set_ylim((0, 0.5))

    s.set_aspect((5 ** 0.5 - 1) / 2 * mplh.aspect_modifier(s))

    fig.tight_layout(pad=0.3)
    fig.savefig(fname)
