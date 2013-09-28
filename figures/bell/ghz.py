import pycuda
import pycuda.driver as cuda
from pycuda.gpuarray import to_gpu, GPUArray
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.curandom import XORWOWRandomNumberGenerator
from pycuda.characterize import sizeof, has_stack
from pycuda.compiler import SourceModule

import numpy
import time
from multiprocessing import Process, Queue

import pickle, json

from mako import exceptions
from mako.template import Template

TEMPLATE = Template(filename='ghz.mako')



def calculation(in_queue, out_queue):

    device_num, params = in_queue.get()

    chunk_size = params['chunk_size']
    chunks_num = params['chunks_num']
    particles = params['particles']
    state = params['state']
    representation = params['representation']
    quantities = params['quantities']

    decoherence = params['decoherence']
    if decoherence is not None:
        decoherence_steps = decoherence['steps']
        decoherence_coeff = decoherence['coeff']
    else:
        decoherence_steps = 0
        decoherence_coeff = 1

    binning = params['binning']
    if binning is not None:
        s = set()
        for names, _, _ in binning:
            s.update(names)
        quantities = sorted(list(s))

    c_dtype = numpy.complex128
    c_ctype = 'double2'
    s_dtype = numpy.float64
    s_ctype = 'double'
    Fs = []

    cuda.init()

    device = cuda.Device(device_num)
    ctx = device.make_context()
    free, total = cuda.mem_get_info()
    max_chunk_size = float(total) / len(quantities) / numpy.dtype(c_dtype).itemsize / 1.1
    max_chunk_size = 10 ** int(numpy.log(max_chunk_size) / numpy.log(10))
    #print free, total, max_chunk_size

    if max_chunk_size > chunk_size:
        subchunk_size = chunk_size
        subchunks_num = 1
    else:
        assert chunk_size % max_chunk_size == 0
        subchunk_size = max_chunk_size
        subchunks_num = chunk_size / subchunk_size

    buffers = []
    for quantity in sorted(quantities):
        buffers.append(GPUArray(subchunk_size, c_dtype))

    stream = cuda.Stream()

    # compile code
    try:
        source = TEMPLATE.render(
            c_ctype=c_ctype, s_ctype=s_ctype, particles=particles,
            state=state, representation=representation, quantities=quantities,
            decoherence_coeff=decoherence_coeff)
    except:
        print exceptions.text_error_template().render()
        raise

    try:
        module = SourceModule(source, no_extern_c=True)
    except:
        for i, l in enumerate(source.split("\n")):
            print i + 1, ":", l
        raise

    kernel_initialize = module.get_function("initialize")
    kernel_calculate = module.get_function("calculate")
    kernel_decoherence = module.get_function("decoherence")

    # prepare call parameters

    gen_block_size = min(
        kernel_initialize.max_threads_per_block,
        kernel_calculate.max_threads_per_block)
    gen_grid_size = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
    gen_block = (gen_block_size, 1, 1)
    gen_grid = (gen_grid_size, 1, 1)

    num_gen = gen_block_size * gen_grid_size
    assert num_gen <= 20000

    # prepare RNG states

    #seeds = to_gpu(numpy.ones(size, dtype=numpy.uint32))
    seeds = to_gpu(numpy.random.randint(0, 2**32 - 1, size=num_gen).astype(numpy.uint32))
    state_type_size = sizeof("curandStateXORWOW", "#include <curand_kernel.h>")
    states = cuda.mem_alloc(num_gen * state_type_size)

    #prev_stack_size = cuda.Context.get_limit(cuda.limit.STACK_SIZE)
    #cuda.Context.set_limit(cuda.limit.STACK_SIZE, 1<<14) # 16k
    kernel_initialize(states, seeds.gpudata, block=gen_block, grid=gen_grid, stream=stream)
    #cuda.Context.set_limit(cuda.limit.STACK_SIZE, prev_stack_size)

    # run calculation
    args = [states] + [buf.gpudata for buf in buffers] + [numpy.int32(subchunk_size)]

    if binning is None:

        results = {quantity:numpy.zeros((decoherence_steps+1, chunks_num * subchunks_num), c_dtype)
            for quantity in quantities}
        for i in xrange(chunks_num * subchunks_num):
            kernel_calculate(*args, block=gen_block, grid=gen_grid, stream=stream)

            for k in xrange(decoherence_steps + 1):
                if k > 0:
                    kernel_decoherence(*args, block=gen_block, grid=gen_grid, stream=stream)

                for j, quantity in enumerate(sorted(quantities)):
                    F = (gpuarray.sum(buffers[j], stream=stream) / buffers[j].size).get()
                    results[quantity][k, i] = F

        for quantity in sorted(quantities):
            results[quantity] = results[quantity].reshape(
                decoherence_steps + 1, chunks_num, subchunks_num).mean(2).real.tolist()

        out_queue.put(results)

    else:

        bin_accums = [numpy.zeros(tuple([binnum] * len(vals)), numpy.int64)
            for vals, binnum, _ in binning]
        bin_edges = [None] * len(binning)

        for i in xrange(chunks_num * subchunks_num):
            bin_edges = []
            kernel_calculate(*args, block=gen_block, grid=gen_grid, stream=stream)
            results = {quantity:buffers[j].get().real for j, quantity in enumerate(sorted(quantities))}

            for binparam, bin_accum in zip(binning, bin_accums):
                qnames, binnum, ranges = binparam
                sample_lines = [results[quantity] for quantity in qnames]
                sample = numpy.concatenate([arr.reshape(subchunk_size, 1) for arr in sample_lines], axis=1)

                hist, edges = numpy.histogramdd(sample, binnum, ranges)
                bin_accum += hist
                bin_edges.append(numpy.array(edges))

        results = [[acc.tolist(), edges.tolist()] for acc, edges in zip(bin_accums, bin_edges)]

        out_queue.put(results)

    #ctx.pop()
    ctx.detach()


cuda.init()

# pick only Teslas
DEVICE_NUMS = []
for i in xrange(cuda.Device.count()):
    if 'Tesla' in cuda.Device(i).name():
        DEVICE_NUMS.append(i)
#DEVICE_NUMS = [DEVICE_NUMS[0]]


def write_log(*args):
    msg = " ".join([str(x) for x in args])
    with open('ghz.txt', 'a+') as f:
        f.write(msg + "\n")
    print ">", msg


def get_quantities(chunk_size, chunks_num, particles, state, representation, quantities,
        decoherence=None, binning=None):

    chunks_nums = [chunks_num / len(DEVICE_NUMS)] * len(DEVICE_NUMS)
    if chunks_num % len(DEVICE_NUMS) != 0:
        for i in range(chunks_num % len(DEVICE_NUMS)):
            chunks_nums[i] += 1

    in_queues = []
    out_queues = []
    processes = []

    params = dict(
        chunk_size=chunk_size,
        particles=particles, state=state, representation=representation, quantities=quantities,
        decoherence=decoherence, binning=binning)

    for i, device_num in enumerate(DEVICE_NUMS):
        in_queue = Queue()
        out_queue = Queue()
        process = Process(target=calculation, args=(in_queue, out_queue))

        processes.append(process)
        in_queues.append(in_queue)
        out_queues.append(out_queue)

        process.start()
        p = dict(params)
        p['chunks_num'] = chunks_nums[i]
        in_queue.put((device_num, p))

    raw_results = []
    for process, queue in zip(processes, out_queues):
        raw_results.append(queue.get())
        process.join()


    if binning is None:

        results = {quantity:list() for quantity in quantities}
        for rres in raw_results:
            for quantity in sorted(quantities):
                results[quantity].append(numpy.array(rres[quantity]))

        processed_results = []
        for quantity in results:
            data = numpy.concatenate(results[quantity], axis=1)
            val, err = mean_and_error(data)
            res = dict(particles=particles, representation=representation,
                state=state, size=chunk_size, subsets=chunks_num, quantity=quantity,
                mean=val.tolist(), error=err.tolist())

            if decoherence is None:
                res['mean'] = res['mean'][0]
                res['error'] = res['error'][0]

            write_log(repr(res) + ",")
            processed_results.append(res)

        return processed_results

    else:

        results = [[numpy.array(acc), numpy.array(edges)] for acc, edges in raw_results[0]]

        for rres in raw_results[1:]:
            for i in xrange(len(rres)):
                results[i][0] += numpy.array(rres[i][0])

        return results


def getF_analytical(particles, state):
    """
    Returns 'classical' and 'quantum' predictions for the
    Mermin's/Ardehali's state and operator.
    """
    if state == 'mermin':
        return 2. ** (particles / 2), 2. ** (particles - 1)
    else:
        return 2. ** ((particles + 1) / 2), 2. ** (particles - 0.5)


def mean_and_error(arr):
    arr = numpy.array(arr)
    return arr.mean(-1), arr.std(-1) / numpy.sqrt(arr.shape[-1])


def collect_binning():
    """Bin distributions for the parts of 2-particle Bell inequality."""

    binning = [
            (['sigma1x', 'sigma2y'], 60, [(-3.5, 3.5), (-3.5, 3.5)]),
            (['sigma1x2y', 'sigma1y2x'], 60, [(-8.5, 6.5), (-6.5, 8.5)]),
        ]

    results = get_quantities(10 ** 6, 10 ** 2, 2, 'ardehali', 'Q', [],
        binning=binning)

    with open('ghz_binning_ardehali_2p_Q.pickle', 'w') as f:
        pickle.dump((binning, results), f, protocol=2)

    results = get_quantities(10 ** 6, 10 ** 2, 2, 'ardehali', 'number', [],
        binning=binning)

    with open('ghz_binning_ardehali_2p_number.pickle', 'w') as f:
        pickle.dump((binning, results), f, protocol=2)


def collect_decoherence():
    """Get results for the superdecoherence."""

    datasets = []
    state = 'ardehali'
    representation = 'Q'

    for particles in (2, 3, 4, 6):
        quantities = ['N_total']
        datasets += get_quantities(10**5, 10**3, particles, state, representation, quantities,
            decoherence=dict(steps=400, coeff=0.1))

    json.dump(datasets, open('ghz_decoherence.json', 'w'))


def collect_violations():
    """Get F values for GHZ states."""

    datasets = []
    representation = 'number'
    for size in (10 ** 9,):
        for particles in range(1, 20):
            state = 'ardehali' if particles % 2 == 0 else 'mermin'
            quantities = ['F_' + state,
                #'N_total', 'max_order_corr'
                ]
            results = get_quantities(size, 10**3, particles, state, representation, quantities)
            datasets += results

            cl, qm = getF_analytical(particles, state)
            if results[0]['error'] / qm > 0.5:
                break

            with open('ghz_violations.json', 'w') as f:
                json.dump(datasets, f, indent=4)



if __name__ == '__main__':

    write_log("\n" + time.ctime() + "\n")

    #collect_decoherence()
    collect_violations()
    #collect_binning()
