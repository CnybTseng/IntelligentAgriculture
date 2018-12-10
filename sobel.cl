__kernel void sobel(__read_only image2d_t in,
                    __write_only image2d_t out,
					__global int *sx,
					__global int *sy)
{
	const sampler_t sampler = CLK_ADDRESS_CLAMP |
                              CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE;
	
	int2 coord = (int2)(get_global_id(0), get_global_id(1));
	uint4 val11 = read_imageui(in, sampler, coord);
	
	int2 dim = get_image_dim(in);
	if (coord.x > 0 && coord.x < dim.x - 1 &&
        coord.y > 0 && coord.y < dim.y - 1)
	{
		uint4 val00 = read_imageui(in, sampler, (int2)(coord.x - 1, coord.y - 1));
		uint4 val01 = read_imageui(in, sampler, (int2)(coord.x,     coord.y - 1));
		uint4 val02 = read_imageui(in, sampler, (int2)(coord.x + 1, coord.y - 1));
		uint4 val10 = read_imageui(in, sampler, (int2)(coord.x - 1, coord.y));
		uint4 val12 = read_imageui(in, sampler, (int2)(coord.x + 1, coord.y));
		uint4 val20 = read_imageui(in, sampler, (int2)(coord.x - 1, coord.y + 1));
		uint4 val21 = read_imageui(in, sampler, (int2)(coord.x,     coord.y + 1));
		uint4 val22 = read_imageui(in, sampler, (int2)(coord.x + 1, coord.y + 1));
		
		short bdx = val00.x * sx[0] + val01.x * sx[1] + val02.x * sx[2] +\
                    val10.x * sx[3] + val11.x * sx[4] + val12.x * sx[5] +\
				    val20.x * sx[6] + val21.x * sx[7] + val22.x * sx[8];
		short bdy = val00.x * sy[0] + val01.x * sy[1] + val02.x * sy[2] +\
                    val10.x * sy[3] + val11.x * sy[4] + val12.x * sy[5] +\
				    val20.x * sy[6] + val21.x * sy[7] + val22.x * sy[8];

		short gdx = val00.y * sx[0] + val01.y * sx[1] + val02.y * sx[2] +\
                    val10.y * sx[3] + val11.y * sx[4] + val12.y * sx[5] +\
				    val20.y * sx[6] + val21.y * sx[7] + val22.y * sx[8];
		short gdy = val00.y * sy[0] + val01.y * sy[1] + val02.y * sy[2] +\
                    val10.y * sy[3] + val11.y * sy[4] + val12.y * sy[5] +\
				    val20.y * sy[6] + val21.y * sy[7] + val22.y * sy[8];

		short rdx = val00.z * sx[0] + val01.z * sx[1] + val02.z * sx[2] +\
                    val10.z * sx[3] + val11.z * sx[4] + val12.z * sx[5] +\
				    val20.z * sx[6] + val21.z * sx[7] + val22.z * sx[8];
		short rdy = val00.z * sy[0] + val01.z * sy[1] + val02.z * sy[2] +\
                    val10.z * sy[3] + val11.z * sy[4] + val12.z * sy[5] +\
				    val20.z * sy[6] + val21.z * sy[7] + val22.z * sy[8];
		
		val11.x = sqrt((float)bdx * bdx + bdy + bdy);
		val11.y = sqrt((float)gdx * gdx + gdy * gdy);
		val11.z = sqrt((float)rdx * rdx + rdy * rdy);
	}
	
	write_imageui(out, coord, val11);
}