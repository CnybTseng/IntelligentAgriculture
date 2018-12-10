__kernel void vecadd(__global const float *a,
                     __global const float *b,
					 __global float *restrict c,
					 unsigned int n)
{
	size_t id = get_global_id(0);
	if (id < n)
		c[id] = a[id] + b[id];
}