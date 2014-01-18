import json
import pickle
import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tests = [
    ((1, 1, 1), (8,8,64), None),

    ((1, 1, 1), (8,8,128), 'axial2_box1'),
    ((1, 1, 2), (8,8,128), 'axial2_box2'),
    ((1, 1, 1.5), (8,8,64), 'axial1_box1.5'),
    ((1, 1, 1.5), (8,8,128), 'axial2_box1.5'),
    ((1, 1, 1.1), (8,8,64), 'axial1_box1.1'),
    ((1, 1, 1.9), (8,8,128), 'axial2_box1.9'),
    ((1, 1, 1.2), (8,8,64), 'axial1_box1.2'),
    ((1, 1, 1.8), (8,8,128), 'axial2_box1.8'),

    ((1, 1, 1), (16,8,64), 'radial2_box1'),
    ((2, 1, 1), (16,8,64), 'radial2_box2'),
    ((1.5, 1, 1), (8,8,64), 'radial1_box1.5'),
    ((1.5, 1, 1), (16,8,64), 'radial2_box1.5'),
    ((1.1, 1, 1), (8,8,64), 'radial1_box1.1'),
    ((1.9, 1, 1), (16,8,64), 'radial2_box1.9'),
    ((1.2, 1, 1), (8,8,64), 'radial1_box1.2'),
    ((1.8, 1, 1), (16,8,64), 'radial2_box1.8'),
    ]


processed = {}

for wigner in (False, True):

    axial_points = {}
    radial_points = {}

    for box_modifiers, shape, suffix in tests:
        prefix = 'ramsey_' + ('wigner' if wigner else 'gpe') + '_test'
        fname = prefix + ('_' + suffix if suffix is not None else '') + '.pickle'
        with open(fname, 'rb') as f:
            data = pickle.load(f)

        V = data['result']['V']['mean']
        assert data['weak_errors']['V'] < 1e-2
        assert data['result']['V']['stderr'][-1] < 1e-2

        axial_spacing = data['box'][2] / data['shape'][2]
        radial_spacing = data['box'][0] / data['shape'][0]

        if suffix is None:
            ref_data = data
            ref_V = V
            ref_V_norm = numpy.linalg.norm(V)
            ref_axial_spacing = axial_spacing
            ref_radial_spacing = radial_spacing

        else:
            rel_axial_spacing = axial_spacing / ref_axial_spacing
            rel_radial_spacing = radial_spacing / ref_radial_spacing
            delta_V = numpy.linalg.norm(V - ref_V) / ref_V_norm

            if rel_axial_spacing != 1.0 or data['box'][2] != ref_data['box'][2]:
                axial_points[rel_axial_spacing] = delta_V
            if rel_radial_spacing != 1.0 or data['box'][0] != ref_data['box'][0]:
                radial_points[rel_radial_spacing] = delta_V

        """
        if suffix is None:
            fig = plt.figure()
            s = fig.add_subplot(111)
            s.set_xlim(0, 1.3)
            s.set_ylim(0, 1)
            s.plot(data['result']['V']['time'], data['result']['V']['mean'])
            fig.savefig(prefix + ".pdf")
        """

    processed['wigner' if wigner else 'gpe'] = dict(
        axial_points=axial_points, radial_points=radial_points)

with open('grid_test.json', 'w') as f:
    json.dump(processed, f, indent=4)
