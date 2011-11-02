// nvcc main.cpp kernels.cu -lglut –lGLU
#include <cmath>
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <GL/glut.h>
#include "kernels.h"
GLuint pointsVBO;
float2 *pointsVBO_device;		// device pointer for VBO
struct cudaGraphicsResource *pointsVBO_Resource;
float2 points[N];
float2 velocities[N];
void init(void)
{
	// GLEW library
	GLenum err = glewInit();
	if (GLEW_OK != err)		exit(0);
	// points
	for(int k=0; k<N; k++)
	{
		#define SCALE 0.5
		points[k] =		make_float2( SCALE*(rand()/float(RAND_MAX)-0.5), SCALE*(rand()/float(RAND_MAX)-0.5) ); 
		velocities[k] =	make_float2(0,0);//0.01*sin(10*points[k].x), 0.01*sin(2*points[k].y) );
	}
	// copy velocity -> device
	void **v_device_addr;
	cudaGetSymbolAddress ((void **)&v_device_addr, "v_device");
	cudaMemcpy (v_device_addr, velocities, sizeof(float2) * N,  cudaMemcpyHostToDevice );
	// ------------ VBO
	// 1. generate vbo 2. activate (hook) 3. upload
	glGenBuffers(1,&pointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STREAM_DRAW);	
	// ------------ CUDA
	cudaGraphicsGLRegisterBuffer(&pointsVBO_Resource, pointsVBO, cudaGraphicsMapFlagsNone );
	// OpenGL
	glEnable(GL_BLEND);	// http://www.pcworld.pl/artykuly/44799/Przezroczystosc.i.mgla.html
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);		
}
void changeSize(int w, int h) 
{
	float ratio = 1.0 * w / h;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, w, h);
	gluOrtho2D(-1,1,-1,1);
	glMatrixMode(GL_MODELVIEW);
}
void renderScene(void) 
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glColor4f(1,1,1,0.7);
	glPointSize(2.0);
	// draw vbo
	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, NULL);  
	glDrawArrays(GL_POINTS, 0, N);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();
}
void idleFunction(void)
{
	// -- timer
	cudaEvent_t start, stop; 
	float time; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	// - run kernel on VBO
	cudaGraphicsMapResources(1, &pointsVBO_Resource, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&pointsVBO_device, &num_bytes, pointsVBO_Resource);
		const float DT = 0.002f;
		call_movepar_VBO(pointsVBO_device, DT);						
	cudaGraphicsUnmapResources(1, &pointsVBO_Resource, 0);
	// -- timer
	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop ); 
	cudaEventDestroy( start ); 
	cudaEventDestroy( stop );
	// -- timer
	#define MILISEC 10000.0f
	printf("%f\n",time/MILISEC);
	glutPostRedisplay();
}
void keyFunction(unsigned char key, int x, int y) {

	if (key == 27)
		exit(0);
}
int main(int argc, char **argv)
{
	cudaGLSetGLDevice(0);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(820,820);
	glutCreateWindow("CUDA points");
	init();
	// register callbacks
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(idleFunction);
	glutKeyboardFunc(keyFunction);
	// enter GLUT event processing cycle
	glutMainLoop();
	cudaGraphicsUnregisterResource(pointsVBO_Resource);
	glDeleteBuffers(1, &pointsVBO);
}