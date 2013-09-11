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
                r"\usepackage[oldstyle]{libertine}", # Linux Libertine fonts for text
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmathfont{Tex Gyre Pagella Math}", # Tex Gyre as main math font
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
import os.path


class AttrDict(defaultdict):

    def __init__(self):
        return defaultdict.__init__(self, lambda: AttrDict())

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def load_colors():
    import xml.etree.ElementTree as ET
    tree = ET.parse(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colors.xml'))
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


dash = {
    '-': [],
    '--': (6, 3),
    '-.': (5, 3, 1, 3),
    ':': (0.5, 2)
}


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

def aspect_modifier(s):
    ys = s.get_ylim()
    xs = s.get_xlim()
    return (xs[1] - xs[0]) / (ys[1] - ys[0])

