import matplotlib as mpl

def setup_backend(draft=False):
    """
    Set rendering backend.
    Final version is rendered with LuaLaTeX and custom text/math fonts ---
    slow, but it's worth it.
    Draft version is rendered with Agg and matplotlib's internal TeX subset.
    """

    if draft:
        backend_name = "Agg"
        backend_params = {
            'backend': 'ps',
            'text.usetex': False
        }
    else:
        backend_name = "pgf"
        backend_params = {
            "text.usetex": True,
            "pgf.texsystem": "lualatex",
            "pgf.rcfonts": False, # don't setup fonts from rc parameters
            "pgf.preamble": [
                # preamble copied from fonts.tex
                r"\usepackage{fontspec}",
                r"\defaultfontfeatures{Ligatures=TeX, Numbers={OldStyle,Proportional}, " +
                    r"Scale=MatchLowercase, Mapping=tex-text}"
                r"\usepackage{libertine}", # Linux Libertine fonts for text
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmathfont{TG Pagella Math}", # Tex Gyre as main math font
                r"\setmathfont[range={\mathcal,\mathbfcal},StylisticSet=1]{Latin Modern Math}"
                ]
        }

    mpl.use(backend_name)
    mpl.rcParams.update(backend_params)


setup_backend() # have to call it before importing pyplot
import matplotlib.pyplot as plt
import numpy
from collections import defaultdict
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


class AttrDict(defaultdict):

    def __init__(self):
        return defaultdict.__init__(self, lambda: AttrDict())

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def load_colors():
    import xml.etree.ElementTree as ET
    tree = ET.parse('colors.xml')
    root = tree.getroot()
    result = AttrDict()

    names = {'primary': 'red', 'secondary-a': 'yellow',
        'secondary-b': 'blue', 'complement': 'green'}
    types = ['main', 'dark', 'darkest', 'light', 'lightest']

    for elem in root:
        if elem.tag != 'colorset':
            continue

        name = elem.attrib['id']
        r = result[names[name]]

        for color in elem:
            ctype = types[int(color.attrib['id'][-1]) - 1]
            r[ctype] = tuple(int(color.attrib[c]) for c in ['r', 'g', 'b'])

    f = AttrDict()
    for color in result:
        for ctype in result[color]:
            f[color][ctype] = tuple(c / 255. for c in result[color][ctype])
    result.f = f

    return result

color = load_colors()


cm_negpos = LinearSegmentedColormap.from_list(
    "negpos",
    [
        color.f.blue.dark,
        color.f.blue.main, color.f.blue.light,
        (1., 1., 1.),
        color.f.red.light, color.f.red.main,
        color.f.red.dark]
    )

cm_zeropos = LinearSegmentedColormap.from_list(
    "zeropos",
    [
        (0.0, (1.0, 1.0, 1.0)),
        (2.5/15, color.f.blue.main),
        (5.0/15, (117/255., 192/255., 235/255.)),
        (10.0/15, color.f.yellow.light),
        (12.5/15, color.f.red.main),
        (1.0, color.f.red.dark)
    ]
    )
cm_zeropos.set_under(color='white')


def setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        'font.size': 10,
        'lines.linewidth': 1.0,
        'lines.dash_capstyle': 'round',

        'legend.fontsize': 'medium',

        # axes
        'axes.labelsize': 10,
        'axes.linewidth': 0.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.pad': 3,
        'xtick.minor.pad': 3,
        'ytick.major.pad': 3,
        'ytick.minor.pad': 3,

        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
    })

setup_style()


def figure(width=0.5, aspect=None):
        column_width_inches = (21. - 2. - 4.) / 2.54 # paper width minus margins, in inches
        if aspect is None:
            aspect = (numpy.sqrt(5) - 1) / 2

        fig_width = column_width_inches * width
        fig_height = fig_width * aspect # height in inches

        return plt.figure(figsize=[fig_width, fig_height])



if __name__ == '__main__':

    """
    f = plt.figure(figsize=(10,5))
    f.subplots_adjust(top=0.8,bottom=0.05,left=0.01,right=0.99)
    maps=[m for m in cm.datad if not m.endswith("_r")] + [cm_negpos, cm_zeropos]
    maps.sort()
    l=len(maps)+1
    a=numpy.outer(numpy.arange(0,1,0.01),numpy.ones(10))
    for i, m in enumerate(maps):
        s=f.add_subplot(1,l,i+1)
        s.axis("off")
        s.imshow(a,aspect='auto',cmap=plt.get_cmap(m),origin="lower")
        s.set_title(m,rotation=90,fontsize=10)
    f.savefig("colormaps.png",dpi=100,facecolor='gray')
    """

    """
    fig = figure()
    s = fig.add_subplot(111)
    s.set_ylim((0, 5))
    s.plot(numpy.ones(5) * 1, color=color.f.red.main)
    s.plot(numpy.ones(5) * 1.5, color=color.f.red.dark)
    s.plot(numpy.ones(5) * 2, color=color.f.blue.main)
    s.plot(numpy.ones(5) * 2.5, color=color.f.blue.dark)
    s.plot(numpy.ones(5) * 3, color=color.f.green.main)
    s.plot(numpy.ones(5) * 3.5, color=color.f.green.dark)
    s.plot(numpy.ones(5) * 4, color=color.f.yellow.main)
    s.plot(numpy.ones(5) * 4.5, color=color.f.yellow.dark)

    s.set_xlabel("unicode text: m")
    s.set_ylabel("XeLaTeX")
    #s.legend(["4 unicode math: $\\mathrm{unicode math}"])
    fig.tight_layout()
    fig.savefig('plot.pdf')
    """
    x, y = numpy.mgrid[-2:2:50j,-2:2:50j]
    """
    fig = figure()
    s = fig.add_subplot(111)
    s.set_xlim((-2, 2))
    s.set_ylim((-2, 2))
    s.imshow(x * numpy.exp(-x ** 2 - y ** 2), cmap=cm_negpos, extent=(-2, 2, -2, 2))
    fig.tight_layout()
    fig.savefig('plot_negpos.pdf')
    """

    fig = figure()
    s = fig.add_subplot(111)
    s.set_xlim((-2, 2))
    s.set_ylim((-2, 2))
    s.imshow(numpy.exp(-x ** 2 - y ** 2), cmap=cm_zeropos, extent=(-2, 2, -2, 2))
    fig.tight_layout()
    fig.savefig('plot_zeropos.pdf')
