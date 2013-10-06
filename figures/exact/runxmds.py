from subprocess import call
import shutil
import os.path
from correlations import processData

def run(executable, fname, **params):
	args = ["--" + name + "=" + str(params[name]) for name in sorted(params.keys())]
	print "*", executable, " ".join(args)
	FNULL = open('/dev/null', 'w')
	retcode = call(["./" + executable] + args, stdout=FNULL, stderr=FNULL)
	if retcode != 0:
		print "!!! Error:", retcode
	else:
		if not os.path.exists('results'):
			os.mkdir('results')
		shutil.move(executable + '.xsil', fname)

def result(executable, **params):
	tag = " ".join([key + "=" + str(params[key]) for key in sorted(params)])
	return os.path.join('results', executable + ' ' + tag + '.xsil')

def resultn(executable, n, **params):
	tag = " ".join([key + "=" + str(params[key]) for key in sorted(params)])
	return os.path.join('results', executable + ' ' + tag + '.' + str(n) + '.xsil')


def get(executable, **params):
	res_file = result(executable, **params)
	if not os.path.exists(res_file):
		run(executable, res_file, **params)
	return processData(res_file)


def deln(executable, n, **params):
	for i in range(n):
		res_file = resultn(executable, i, **params)
		if os.path.exists(res_file):
			os.remove(res_file)


def getn(executable, n, **params):
	datasets = []
	for i in range(n):
		print "--- Getting", i, "-th set of results ---"
		res_file = resultn(executable, i, **params)
		if not os.path.exists(res_file):
			run(executable, res_file, **params)
		datasets.append(processData(res_file))
	return datasets
