
#include "device_launch_parameters.h"

#include "kernel.h"

__global__ void kernel(uchar4 *ptr, GLfloat min, GLfloat max)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	/*float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;

	unsigned char red = 128 + 127 * sin(fabs(fx * 100) - fabs(fy * 100));
	unsigned char green = 128 + 127 * sin(fabs(fx * 100) - fabs(fy * 100));
	unsigned char blue = 128 + 127 * sin(fabs(fx * 100) - fabs(fy * 100));*/

	GLfloat a, b, color;
	a = min + ((GLfloat)x / DIM * (max - min));
	b = min + ((GLfloat)y / DIM * (max - min));

	GLfloat ca = a;
	GLfloat cb = b;

	/*GLfloat ca = 0.36024;
	GLfloat cb = -0.64131;*/

	//GLfloat ca = 0.0;
	//GLfloat cb = -0.8;

	/* Check for divergence */
	int iter = 0;
	for (iter = 0; iter < 256; iter++)
	{
		GLfloat real = (a*a) - (b*b);
		GLfloat imag = 2 * a * b;

		a = real + ca;
		b = imag + cb;

		if ((a*a) + (b*b) > 256.0)
		{
			break;
		}
	}

	/* Draw the pixel */

	//color = iter / 256.0f * 256.0f ;

	//if (iter == 100)
	//	color = 0;

	if (iter == 256) iter = 0;

	ptr[offset].x = 0;
	ptr[offset].y = iter;
	ptr[offset].z = 0;
	ptr[offset].w = 255;

	/*ptr[offset].x = red * _red;
	ptr[offset].y = green * _green;
	ptr[offset].z = blue * _blue;
	ptr[offset].w = 255;*/

}

void MandelbrotSet(dim3 grids, dim3 threads, uchar4 *devPtr, GLfloat min, GLfloat max)
{
	kernel<<<grids, threads>>>(devPtr, min, max);
}

__global__ void map(GLfloat num, GLfloat min, GLfloat max, GLfloat newMin, GLfloat newMax, GLfloat *ans)
{
	*ans = newMin + (num / (max - min) * (newMax - newMin));
}