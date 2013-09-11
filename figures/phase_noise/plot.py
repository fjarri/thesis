import numpy
import json

from figures import get_path
import figures.mplhelpers as mplh


def ramsey_short(fname):
    with open(get_path(__file__, 'visibility/ramsey_experimental.json')) as f:
        vis_exp = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_icols2011.json')) as f:
        icols2011 = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_gpe.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_wigner.json')) as f:
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
    with open(get_path(__file__, 'visibility/echo_gpe.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/echo_wigner.json')) as f:
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
    with open(get_path(__file__, 'visibility/ramsey_long_gpe.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/ramsey_long_wigner.json')) as f:
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
    with open(get_path(__file__, 'visibility/echo_long_gpe.json')) as f:
        gpe = json.load(f)
    with open(get_path(__file__, 'visibility/echo_long_wigner.json')) as f:
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


def ramsey_noise(fname):
    pass


def spinecho_noise(fname):
    pass
