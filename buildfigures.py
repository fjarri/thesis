from beclab import *


if __name__ == '__main__':

	# Ground states

	for name in ('1k', '100k'):
		gs = XYData.load('data/mean_field/ground_states/gs_' + name + '.json')
		tf_gs = XYData.load('data/mean_field/ground_states/tf_gs_' + name + '.json')
		XYPlot([gs, tf_gs]).save('figures_generated/mean_field/ground_states_' + name + '.eps')

	gs_a_two_comp = XYData.load('data/mean_field/two_comp_gs/gs_a_80k.json')
	gs_b_two_comp = XYData.load('data/mean_field/two_comp_gs/gs_b_80k.json')
	XYPlot([gs_a_two_comp, gs_b_two_comp]).save('figures_generated/mean_field/two_comp_gs.eps')
