#define GL_GLEXT_PROTOTYPES
#define _USE_MATH_DEFINES 1

// Headers
#include <Windows.h>
#include <stdio.h>
#include <GL/glew.h>
#include <GL/GLU.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "kernel.h"

// Linker Options
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "opengl32.lib")

/* Shared Data */
GLuint bufferObj = 0;
cudaGraphicsResource *resource = NULL;

// Global Variables
bool gbActiveWindow = false;
bool gbIsFullScreen = true;
FILE *gpFile = NULL;
HDC ghDC = NULL;
HGLRC ghRC = NULL;
HWND ghWnd = NULL;
DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

GLfloat gHeight, gWidth;
uchar4* devPtr;
dim3 grids(DIM / 16, DIM / 16);
dim3 threads(16, 16);

GLfloat min = -2.5f, max = 2.5f;

// Global function declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

#pragma region Window
// WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	// function declarations
	int initialize(void);
	void display(void);
	void update(void);
	void FullScreen(void);

	// variables 
	bool bDone = false;
	int iRet = 0;
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szClassName[] = TEXT("MyApp");

	// code
	// create file for logging
	if (fopen_s(&gpFile, "log.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Cannot Create log file!"), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	else
	{
		fprintf(gpFile, "Log.txt file created...\n");
	}

	/* CUDA */
	cudaDeviceProp prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;

	cudaChooseDevice(&dev, &prop);

	cudaGLSetGLDevice(dev);

	// initialization of WNDCLASSEX
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szClassName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

	// register class
	RegisterClassEx(&wndclass);

	// create window
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szClassName,
		TEXT("Dynamic India"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		100,
		100,
		DIM,
		DIM,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghWnd = hwnd;
	
	iRet = initialize();
	if (iRet == -1)
	{
		fprintf(gpFile, "ChoosePixelFormat failed...\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -2)
	{
		fprintf(gpFile, "SetPixelFormat failed...\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -3)
	{
		fprintf(gpFile, "wglCreateContext failed...\n");
		DestroyWindow(hwnd);
	}
	else if (iRet == -4)
	{
		fprintf(gpFile, "wglMakeCurrent failed...\n");
		DestroyWindow(hwnd);
	}
	else
	{
		fprintf(gpFile, "initialize() successful...\n");
	}

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);
	FullScreen();

	glewInit();

	/* CUDA */
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
	

	// Game Loop 
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				bDone = true;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				update();
			}

			display();
		}
	}

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	// function declarations
	void resize(int, int);
	void uninitialize();

	// code
	switch (iMsg)
	{
	case WM_CREATE:
		break;

	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_UP:
			max /= 1.1;
			min = -max;
			break;

		case VK_DOWN:
			max *= 1.1;
			min = -max;
			break;

		case VK_LEFT:
			max -= 0.1;
			min -= 0.1;
			break;

		case VK_RIGHT:
			max += 0.1;
			min += 0.1;
			break;

		case VK_ESCAPE:

			cudaGraphicsUnregisterResource(resource);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			glDeleteBuffers(1, &bufferObj);

			DestroyWindow(hwnd);
			break;
		}
		break;

		// returned from here, to block DefWindowProc
		// We have our own painter
	case WM_ERASEBKGND:
		return(0);
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void FullScreen()
{
	MONITORINFO MI;

	dwStyle = GetWindowLong(ghWnd, GWL_STYLE);
	if (dwStyle & WS_OVERLAPPEDWINDOW)
	{
		MI = { sizeof(MONITORINFO) };
		if (GetWindowPlacement(ghWnd, &wpPrev)
			&& GetMonitorInfo(MonitorFromWindow(ghWnd, MONITORINFOF_PRIMARY), &MI))
		{
			SetWindowLong(ghWnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
			SetWindowPos(ghWnd,
				HWND_TOP,
				MI.rcMonitor.left,
				MI.rcMonitor.top,
				MI.rcMonitor.right - MI.rcMonitor.left,
				MI.rcMonitor.bottom - MI.rcMonitor.top,
				SWP_NOZORDER | SWP_FRAMECHANGED);
		}
	}
	ShowCursor(FALSE);

}

#pragma endregion

#pragma region OpenGL Helpers

int initialize(void)
{
	// function declarations
	void resize(int, int);

	// variable declarations
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	// code
	// initialize pdf structure
	ZeroMemory((void *)&pfd, sizeof(PIXELFORMATDESCRIPTOR));
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;

	ghDC = GetDC(ghWnd);

	iPixelFormatIndex = ChoosePixelFormat(ghDC, &pfd);
	// iPixelFormatIndex is 1 based, so 0 indicates error
	if (iPixelFormatIndex == 0)
	{
		return(-1);
	}

	if (SetPixelFormat(ghDC, iPixelFormatIndex, &pfd) == FALSE)
	{
		return(-2);
	}

	ghRC = wglCreateContext(ghDC);
	if (ghRC == NULL)
	{
		return(-3);
	}

	if (wglMakeCurrent(ghDC, ghRC) == FALSE)
	{
		return(-4);
	}

	// clear the screen by OpenGL
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// warm-up call to resize
	resize(DIM, DIM);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	return(0);
}

void uninitialize(void)
{
	// fullscreen check
	if (gbIsFullScreen == true)
	{
		SetWindowLong(ghWnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghWnd, &wpPrev);
		SetWindowPos(ghWnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

		ShowCursor(TRUE);
	}

	// break the current context
	if (wglGetCurrentContext() == ghRC)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghRC)
	{
		wglDeleteContext(ghRC);
	}

	if (ghDC)
	{
		ReleaseDC(ghWnd, ghDC);
		ghDC = NULL;
	}

	if (gpFile)
	{
		fprintf(gpFile, "Log file is closed...\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}

void display(void)
{
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	SwapBuffers(ghDC);
}

void resize(int width, int height)
{
	if (height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

}

void update(void)
{
	static float color = 1.0;

	size_t size;

	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

	//if(color > 3.0)
	//	MandelbrotSet(grids, threads, devPtr, 1, 0, 0);
	//else if(color > 2.0)
	//	MandelbrotSet(grids, threads, devPtr, 0, 1, 0);
	//else
	//	MandelbrotSet(grids, threads, devPtr, 0, 0, 1);
	
	MandelbrotSet(grids, threads, devPtr, min, max);

	cudaGraphicsUnmapResources(1, &resource, NULL);



	color+= 0.001f;
	if (color > 4.0)
		color = 1.0;
}