<%!
    from itertools import permutations
    import numpy
%>

#include <curand_kernel.h>

<%
    curand_normal2 = "curand_normal2" + ("" if s_ctype == 'float' else "_double")
    curand_normal = "curand_normal" + ("" if s_ctype == 'float' else "_double")
    curand_uniform = "curand_uniform" + ("" if s_ctype == 'float' else "_double")
    state_coeff = "make_" + c_ctype + ("(0, 1)" if state == "mermin" else "(-1, 0)")
    multiplier = 2 if representation == "spin" else 1
%>

%if representation in ("number", "spin"):
#define sigma_to_real(a) ((a).x)
#define sigma_zero make_${c_ctype}(0, 0)
#define sigma_unit make_${c_ctype}(1, 0)
typedef ${c_ctype} sigma_type;
%else:
#define sigma_to_real(a) (a)
#define sigma_zero 0
#define sigma_unit 1
typedef ${s_ctype} sigma_type;
%endif

inline __device__ ${c_ctype} operator+(${c_ctype} a, ${c_ctype} b)
{
    return make_${c_ctype}(a.x + b.x, a.y + b.y);
}

inline __device__ ${c_ctype} operator+(${s_ctype} a, ${c_ctype} b)
{
    return make_${c_ctype}(a + b.x, b.y);
}

inline __device__ ${c_ctype} operator-(${c_ctype} a, ${c_ctype} b)
{
    return make_${c_ctype}(a.x - b.x, a.y - b.y);
}

inline __device__ ${c_ctype} operator+(${c_ctype} a)
{
    return a;
}

inline __device__ ${c_ctype} operator-(${c_ctype} a)
{
    return make_${c_ctype}(-a.x, -a.y);
}

inline __device__ ${c_ctype} operator*(${c_ctype} a, ${c_ctype} b)
{
    return make_${c_ctype}(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

inline __device__ ${c_ctype} operator*(${c_ctype} a, ${s_ctype} b)
{
    return make_${c_ctype}(a.x * b, a.y * b);
}

inline __device__ ${c_ctype} operator/(${c_ctype} a, ${s_ctype} b)
{
    return make_${c_ctype}(a.x / b, a.y / b);
}

inline __device__ ${s_ctype} cabs_squared(${c_ctype} a)
{
    return a.x * a.x + a.y * a.y;
}

inline __device__ ${c_ctype} conj(${c_ctype} a)
{
    return make_${c_ctype}(a.x, -a.y);
}

inline __device__ ${c_ctype} ctranspose(${c_ctype} a)
{
    return make_${c_ctype}(a.y, a.x);
}

inline __device__ ${c_ctype} cexp(${s_ctype} angle)
{
    return make_${c_ctype}(cos(angle), sin(angle));
}


extern "C" {

__global__ void initialize(curandStateXORWOW *states, unsigned int *seeds)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seeds[idx], idx, 0, &states[idx]);
}

__device__ ${s_ctype} genlambda(curandStateXORWOW *state,
    ${c_ctype} &rand_normal, int &normals_idx)
{
    <%
        alpha = 2
        d = alpha - 1. / 3
        c = 1 / numpy.sqrt(9 * d)
    %>
    ${s_ctype} d = ${d};
    ${s_ctype} c = ${c};

    for (;;)
    {
        ${s_ctype} X, V;
        float U;
        do
        {
            if (normals_idx == 0)
            {
                X = rand_normal.x;
                normals_idx = 1;
            }
            else if (normals_idx == 1)
            {
                X = rand_normal.y;
                normals_idx = 2;
            }
            else
            {
                rand_normal = ${curand_normal2}(state);
                X = rand_normal.x;
                normals_idx = 1;
            }

            V = 1.0 + c * X;
        } while (V <= 0.0);

        V = V * V * V;
        U = curand_uniform(state);
        if (U < 1.0 - 0.0331 * (X * X) * (X * X)) return (d * V);
        if (log(U) < 0.5 * X * X + d * (1. - V + log(V))) return (d * V);
    }
}


// calculates Ptarget / (M * Pbound)
__device__ ${s_ctype} relative_P_number(${c_ctype} *sample)
{
    ${c_ctype} product = make_${c_ctype}(1, 0);
    %for i in xrange(particles):
    product = product * sample[${i}];
    %endfor
    ${s_ctype} target = cabs_squared(product + ${state_coeff});
    ${s_ctype} bound = 2 * (cabs_squared(product) + 1);
    return target / bound;
}

// calculates Ptarget / (M * Pbound)
__device__ ${s_ctype} relative_P_spin(${c_ctype} *sample)
{
    ${c_ctype} product1 = make_${c_ctype}(1, 0);
    ${c_ctype} product2 = make_${c_ctype}(1, 0);

    %for i in xrange(particles):
    product1 = product1 * sample[${i}];
    product2 = product2 * sample[${i + particles}];
    %endfor

    ${s_ctype} target = cabs_squared(product1 + conj(${state_coeff}) * product2);
    ${s_ctype} bound = 2 * (cabs_squared(product1) + cabs_squared(product2));
    return target / bound;
}

// calculates Ptarget / (M * Pbound) for Husimi Q
__device__ ${s_ctype} relative_P_Q(${c_ctype} *sample)
{
    ${c_ctype} product = make_${c_ctype}(1, 0);

    %for i in xrange(particles):
    product = product * sample[${i}];
    %endfor

    ${s_ctype} target = cabs_squared(conj(product) + ${state_coeff});
    ${s_ctype} bound = 2 * (cabs_squared(product) + 1);
    return target / bound;
}


<%def name="insertComplexSigmas()">

    sigma_type sigma_x[${particles}];
    sigma_type sigma_y[${particles}];

    %if representation in ("number", "spin"):
        ${c_ctype} *alpha = sample;
        ${c_ctype} *beta = sample + ${particles * multiplier};
    %endif

    %for i in xrange(particles):
    %if representation == "number":
        sigma_x[${i}] = alpha[${i}] + beta[${i}];
        // (conj . ctranspose) == /1j
        sigma_y[${i}] = conj(ctranspose(alpha[${i}] - beta[${i}]));
    %elif representation == "spin":
        sigma_x[${i}] =
            beta[${i}] * alpha[${i + particles}] +
            beta[${i + particles}] * alpha[${i}];
        // (conj . ctranspose) == /1j
        sigma_y[${i}] =
            conj(ctranspose(beta[${i}] * alpha[${i + particles}] -
            beta[${i + particles}] * alpha[${i}]));
    %else:
        sigma_x[${i}] = 3 * 2 * sample[${i}].x / (1 + cabs_squared(sample[${i}]));
        sigma_y[${i}] = -3 * 2 * sample[${i}].y / (1 + cabs_squared(sample[${i}]));
    %endif
    %endfor
</%def>


// Number of particles
__device__ ${s_ctype} get_N_total(${c_ctype} *sample)
{
    ${s_ctype} N = 0;

    %for i in xrange(particles):
    %if representation == 'number':
        N += (sample[${i}] * sample[${i + particles}]).x;
    %elif representation == 'spin':
        N += 0.5 * ((sample[${i}] * sample[${i + particles * multiplier}]).x -
            (sample[${i + particles}] * sample[${i + particles + particles * multiplier}]).x) + 0.5;
    %elif representation == 'Q':
    {
        ${s_ctype} z2 = cabs_squared(sample[${i}]);
        N += 1.5 * (z2 - 1) / (1 + z2) + 0.5;
    }
    %endif
    %endfor

    return N;
}


// Average M-th-order correlations, eg \prod_k(\sigma_k_x)
__device__ ${s_ctype} get_max_order_corr(${c_ctype} *sample)
{
    ${insertComplexSigmas()}

    sigma_type res = sigma_unit;
    %for i in xrange(particles):
    sigma_y[${i}] = sigma_y[${i}];
    res = res * sigma_x[${i}];
    %endfor
    return sigma_to_real(res);
}


// calculates Mermin's F
__device__ ${s_ctype} get_F_mermin(${c_ctype} *sample)
{
    ${insertComplexSigmas()}

    ${c_ctype} product = make_${c_ctype}(1, 0);

    %for i in xrange(particles):
    product = product * (sigma_x[${i}] + make_${c_ctype}(0, 1) * sigma_y[${i}]);
    %endfor

    return product.y;
}


// calculates Ardehali's F
__device__ ${s_ctype} get_F_ardehali(${c_ctype} *sample)
{
    ${insertComplexSigmas()}

    ${c_ctype} product = make_${c_ctype}(1, 0);

    %for i in xrange(particles):
    product = product * (sigma_x[${i}] + make_${c_ctype}(0, 1) * sigma_y[${i}]);
    %endfor

    return - ${numpy.sqrt(2)} * product.x;
}


// calculates separate spin values
%for axis, number in [('x', 1), ('y', 1), ('x', 2), ('y', 2)]:
__device__ ${s_ctype} get_sigma${number}${axis}(${c_ctype} *sample)
{
    ${insertComplexSigmas()}

    %if representation == 'Q':
    return sigma_${axis}[${number - 1}];
    %else:
    return sigma_${axis}[${number - 1}].x;
    %endif
}
%endfor

%for axes in (('x', 'x'), ('y', 'y'), ('x', 'y'), ('y', 'x')):
__device__ ${s_ctype} get_sigma1${axes[0]}2${axes[1]}(${c_ctype} *sample)
{
    ${insertComplexSigmas()}

    %if representation == 'Q':
    return sigma_${axes[0]}[0] * sigma_${axes[1]}[1];
    %else:
    return (sigma_${axes[0]}[0] * sigma_${axes[1]}[1]).x;
    %endif
}
%endfor



__global__ void calculate(curandStateXORWOW *states,
    %for quantity in quantities:
    ${c_ctype} *${quantity},
    %endfor
    int samples)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int num_gens = blockDim.x * gridDim.x;

    ${c_ctype} sample[${particles * multiplier * (1 if representation == 'Q' else 2)}];

    int sample_idx = idx;
    curandStateXORWOW state = states[idx];

    while (sample_idx < samples)
    {
        %if representation == "number":

        // generate sample
        if (sample_idx < samples / 2)
        {
            // sampling P1 (gaussian)
            %for i in xrange(particles):
            {
                ${c_ctype} temp = ${curand_normal2}(&state);
                temp.x *= ${numpy.sqrt(0.5)};
                temp.y *= ${numpy.sqrt(0.5)};
                sample[${i}] = temp;
            }
            %endfor
        }
        else
        {
            // sampling P2 (gamma)

            ${c_ctype} rand_normal = ${curand_normal2}(&state);
            int normals_idx = 0;

            %for i in xrange(particles):
            {
                ${s_ctype} phi = ${curand_uniform}(&state) * ${2 * numpy.pi};
                ${s_ctype} radius = sqrt(genlambda(&state, rand_normal, normals_idx));
                ${c_ctype} temp = make_${c_ctype}(cos(phi) * radius, sin(phi) * radius);
                sample[${i}] = temp;
            }
            %endfor
        }

        // check sample
        float reject_decision = curand_uniform(&state);
        ${s_ctype} ratio = relative_P_number(sample);

        %elif representation == 'spin':

        int gaussian_offset, gamma_offset;
        if (sample_idx < samples / 2)
        {
            gaussian_offset = 0;
            gamma_offset = ${particles};
        }
        else
        {
            gaussian_offset = ${particles};
            gamma_offset = 0;
        }

        // sampling P1 (gaussian)
        %for i in xrange(particles):
        {
            ${c_ctype} temp = ${curand_normal2}(&state);
            temp.x *= ${numpy.sqrt(0.5)};
            temp.y *= ${numpy.sqrt(0.5)};
            sample[${i} + gaussian_offset] = temp;
        }
        %endfor

        // sampling P2 (gamma)

        ${c_ctype} rand_normal = ${curand_normal2}(&state);
        int normals_idx = 0;

        %for i in xrange(particles):
        {
            ${s_ctype} phi = ${curand_uniform}(&state) * ${2 * numpy.pi};
            ${s_ctype} radius = sqrt(genlambda(&state, rand_normal, normals_idx));
            ${c_ctype} temp = make_${c_ctype}(cos(phi) * radius, sin(phi) * radius);
            sample[${i} + gamma_offset] = temp;
        }
        %endfor

        // check sample
        float reject_decision = curand_uniform(&state);
        ${s_ctype} ratio = relative_P_spin(sample);

        %else:

        ${s_ctype} u, phi, X;
        %for i in xrange(particles):
            u = ${curand_uniform}(&state);
            phi = ${curand_uniform}(&state) * ${2 * numpy.pi};

            if (sample_idx < samples / 2)
                X = sqrt(sqrt(1 / (1 - u)) - 1);
            else
                X = sqrt(sqrt(u) / (1 - sqrt(u)));

            sample[${i}] = make_${c_ctype}(X * cos(phi), X * sin(phi));
        %endfor

        float reject_decision = curand_uniform(&state);
        ${s_ctype} ratio = relative_P_Q(sample);

        %endif

        if (reject_decision < ratio)
        {
            %if representation in ('number', 'spin'):

                // calculate linear combination with gaussian randoms (mu) and save
                %for i in xrange(particles * multiplier):
                {
                    ${c_ctype} mu = ${curand_normal2}(&state);
                    mu.x *= ${numpy.sqrt(0.5)};
                    mu.y *= ${numpy.sqrt(0.5)};

                    ${c_ctype} lambda = sample[${i}];

                    sample[${i}] = lambda + mu;
                    sample[${i + particles * multiplier}] = conj(lambda - mu);
                }
                %endfor

            %endif

            %for quantity in sorted(quantities):
            ${quantity}[sample_idx] = make_${c_ctype}(get_${quantity}(sample), 0);
            %endfor
            sample_idx += num_gens;
        }
    }

    states[idx] = state;
}


__global__ void decoherence(curandStateXORWOW *states,
    %for quantity in sorted(quantities):
    ${c_ctype} *${quantity},
    %endfor
    int samples)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int num_gens = blockDim.x * gridDim.x;

    int sample_idx = idx;
    curandStateXORWOW state = states[idx];

    ${c_ctype} temp;
    ${s_ctype} random;

    while (sample_idx < samples)
    {
        %for quantity in sorted(quantities):
        temp = ${quantity}[sample_idx];
        random = ${curand_normal}(&state) * ${particles};
        ${quantity}[sample_idx] = temp * cexp(${decoherence_coeff} * random);
        %endfor

        sample_idx += num_gens;
    }

    states[idx] = state;
}

}
