import xml.etree.ElementTree as et
import numpy

class AttrDict(dict):

	def __getattr__(self, attr):
		return self[attr]

	def __setattr__(self, attr, value):
		self[attr] = value


def load(fname):
	tree = et.ElementTree(file=open(fname))
	xsil = tree.getroot().find('XSIL')
	names = []
	data = None

	for elem in xsil:
		if elem.tag == 'Array' and elem.attrib['Name'] == 'variables':
			text = elem.find('Stream').find('Metalink').tail
			names = text.strip().split(' ')
		elif elem.tag == 'Array' and elem.attrib['Name'] == 'data':
			data_raw = elem.find('Stream').find('Metalink').tail.strip()
			data_lists = [[float(x) for x in line.split(' ')] for line in data_raw.split('\n')]
			data = numpy.array(data_lists)

	result = AttrDict()
	for i, name in enumerate(names):
		result[name] = data[:, i]

	return result

def load_means(fname):
	data = load(fname)
	result = AttrDict()
	for key in data.keys():
		if key.startswith('error_') or key.startswith('stderr_'):
			continue
		elif key.startswith('mean_re_'):
			new_key = key[8:]
			if new_key in result:
				result[new_key] = result[new_key] + data[key]
			else:
				result[new_key] = data[key]
		elif key.startswith('mean_im_'):
			new_key = key[8:]
			if new_key in result:
				result[new_key] = result[new_key] + 1j * data[key]
			else:
				result[new_key] = 1j * data[key]
		elif key.startswith('mean_'):
			new_key = key[5:]
			result[new_key] = data[key]
		else:
			result[key] = data[key]
	return result

if __name__ == '__main__':
	data = load_means('sde_4mode.xsil')
	print data.alpha1_calpha2