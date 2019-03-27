#pragma once

#define GL_GLEXT_PROTOTYPES

#include <Windows.h>

#include <math.h>

#include <gl/GL.h>
#include <gl/GLU.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#define DIMX 1920
#define DIMY 1080

void gpuMandelbrotSet(dim3, dim3, uchar4*, GLdouble, GLdouble );
void cpuMandelbrotSet(uchar4*, GLdouble, GLdouble );
//void MandelbrotSet(dim3, dim3, uchar4*, int, int, int);
