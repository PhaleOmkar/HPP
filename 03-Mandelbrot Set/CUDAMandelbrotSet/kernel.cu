
#include "device_launch_parameters.h"

#include "kernel.h"

__global__ void kernel(uchar4 *ptr, GLdouble min, GLdouble max)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < DIMX && y < DIMY)
	{
		int offset = x + y * blockDim.x * gridDim.x;

		GLdouble a, b, color;
		a = min + ((GLdouble)x / DIMX * (max - min));
		b = (min*(GLdouble)DIMY/ (GLdouble)DIMX) + ((GLdouble)y / DIMY * ((max*(GLdouble)DIMY / (GLdouble)DIMX) - (min*(GLdouble)DIMY / (GLdouble)DIMX)));

		GLdouble ca = a;
		GLdouble cb = b;

		/*GLdouble ca = 0.36024;
		GLdouble cb = -0.64131;
*/
		//GLdouble ca = 0.0;
		//GLdouble cb = -0.8;

		/* Check for divergence */
		int iter = 0;
		for (iter = 0; iter < 100; iter++)
		{
			GLdouble real = (a*a) - (b*b);
			GLdouble imag = 2 * a * b;

			a = real + ca;
			b = imag + cb;

			if ((a*a) + (b*b) > 256.0f)
			{
				break;
			}
		}

		/* Draw the pixel */


		if (iter == 100) iter = 0;

		ptr[offset].x = 0;
		ptr[offset].y = (GLdouble)iter / 100.0 * (255.0);
		ptr[offset].z = 0;
		ptr[offset].w = 255;
	}
}

void gpuMandelbrotSet(dim3 grids, dim3 threads, uchar4 *devPtr, GLdouble min, GLdouble max)
{
	kernel<<<grids, threads>>>(devPtr, min, max);
}

void cpuMandelbrotSet(uchar4* ptr, GLdouble min, GLdouble max)
{
	for (int x = 0; x < DIMX; x++)
	{
		for (int y = 0; y < DIMY; y++)
		{
			int offset = x + y * DIMX;

			GLdouble a, b, color;
			a = min + ((GLdouble)x / DIMX * (max - min));
			b = (min*(GLdouble)DIMY / (GLdouble)DIMX) + ((GLdouble)y / DIMY * ((max*(GLdouble)DIMY / (GLdouble)DIMX) - (min*(GLdouble)DIMY / (GLdouble)DIMX)));

			GLdouble ca = a;
			GLdouble cb = b;

			/*GLdouble ca = 0.36024;
			GLdouble cb = -0.64131;*/
			
			//GLdouble ca = 0.0;
			//GLdouble cb = -0.8;

			/* Check for divergence */
			int iter = 0;
			for (iter = 0; iter < 100; iter++)
			{
				GLdouble real = (a*a) - (b*b);
				GLdouble imag = 2 * a * b;

				a = real + ca;
				b = imag + cb;

				if ((a*a) + (b*b) > 256.0f)
				{
					break;
				}
			}

			/* Draw the pixel */
			if (iter == 100) iter = 0;

			ptr[offset].x = 0;
			ptr[offset].y = (GLdouble)iter / 100.0 * (255.0);
			ptr[offset].z = 0;
			ptr[offset].w = 255;
		}
	}
}