#pragma once

#define GL_GLEXT_PROTOTYPES

#include <Windows.h>

#include <math.h>

#include <gl/GL.h>
#include <gl/GLU.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#define DIM 1024

void MandelbrotSet(dim3, dim3, uchar4*, GLfloat, GLfloat );
//void MandelbrotSet(dim3, dim3, uchar4*, int, int, int);
__global__ void map(GLfloat num, GLfloat min, GLfloat max, GLfloat newMin, GLfloat newMax, GLfloat *ans);
