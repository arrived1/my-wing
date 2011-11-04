// nvcc main.cpp kernels.cu -lglut –lGLU
#include <cmath>
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <GL/glut.h>
#include "kernels.h"
#include <parameter.hpp>

GLuint pointsVBO;
float2 *pointsVBO_device; // device pointer for VBO
struct cudaGraphicsResource *pointsVBO_Resource;
float2 points[N];
float2 velocities[N];

void camera();

void init(void)
{
	// GLEW library
	GLenum err = glewInit();
	if (GLEW_OK != err)		
        exit(0);
	
    // points
    for(int k=0; k<N; k++)
    {
        #define SCALE 100 //0.5
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
	
    //// ------------ CUDA
    cudaGraphicsGLRegisterBuffer(&pointsVBO_Resource, pointsVBO, cudaGraphicsMapFlagsNone );
	
    // OpenGL
	glEnable(GL_BLEND);	// http://www.pcworld.pl/artykuly/44799/Przezroczystosc.i.mgla.html
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);		
	
    glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glClearColor(0.0f,0.0f,0.0f,0.0f);					// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glDisable(GL_DEPTH_TEST);							// Disable Depth Testing
	glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST);	// Really Nice Perspective Calculations
	glHint(GL_POINT_SMOOTH_HINT,GL_NICEST);				// Really Nice Point Smoothing
}

void changeSize(int w, int h) 
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0, (float) w / (float) h, 0.1, parameter::a + 400.0);
    glTranslatef(0, 0, -parameter::a);
   
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void renderScene(void) 
{
    camera();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(100);

    glColor3f(1.0, .0, .0);
    glBegin(GL_LINES);
        glVertex3f(.0f, .0f, .0f);  //x
        glVertex3f(parameter::a/2 + 20.0f, .0f, .0f);

        glVertex3f(parameter::a/2 + 20.0f, .0f, .0f);	//strzalka gora
        glVertex3f(parameter::a/2 + 15.0f, 2.0f, .0f);

        glVertex3f(parameter::a/2 + 20.0f, .0f, .0f);	//strzalka dol
        glVertex3f(parameter::a/2 + 15.0f, -2.0f, .0f);

        glVertex3f(.0f, .0f, .0f);  //y
        glVertex3f(.0f, parameter::a/2 + 20.0f, .0f);

        glVertex3f(.0f, parameter::a/2 + 20.0f, .0f);	//strzalka prawo
        glVertex3f(2.0f, parameter::a/2 + 15.0f, .0f);

        glVertex3f(.0f, parameter::a/2 + 20.0f, .0f);	//strzalka lewo
        glVertex3f(-2.0f, parameter::a/2 + 15.0f, .0f);

        glVertex3f(.0f, .0f, .0f);  //z
        glVertex3f(.0f, .0f, parameter::a/2 + 20.0f);

        glVertex3f(.0f, .0f, parameter::a/2 + 20.0f);	//strzalka lewo
        glVertex3f(-2.0f, .0f, parameter::a/2 + 15.0f);

        glVertex3f(.0f, .0f, parameter::a/2 + 20.0f);	//strzalka prawo
        glVertex3f(2.0f, .0f, parameter::a/2 + 15.0f);
    glEnd();

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

void keyFunction(unsigned char key, int, int) 
{
	if (key == 27)
		exit(0);
}

void motion(int x, int y)
{
    float dx = x - parameter::ox;
    float dy = y - parameter::oy;

    if (parameter::buttonState == 3) 
    {
        // left+middle = zoom
        parameter::camera_trans[2] += (dy / 100.0) * 0.5 * fabs(parameter::camera_trans[2]);
    } 
    else if (parameter::buttonState & 2) 
    {
        // middle = translate
        parameter::camera_trans[0] += dx / 10.0;
        parameter::camera_trans[1] -= dy / 10.0;
    }
    else if (parameter::buttonState & 1) 
    {
        // left = rotate
        parameter::camera_rot[0] += dy / 5.0;
        parameter::camera_rot[1] += dx / 5.0;
    }

    parameter::ox = x; 
    parameter::oy = y;

    glutPostRedisplay();
}

void camera()
{
	// view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    for (int c = 0; c < 3; ++c)
    {
        parameter::camera_trans_lag[c] += 
            (parameter::camera_trans[c] - parameter::camera_trans_lag[c]) * parameter::inertia;
        parameter::camera_rot_lag[c] += 
            (parameter::camera_rot[c] - parameter::camera_rot_lag[c]) * parameter::inertia;
    }
    glTranslatef(parameter::camera_trans_lag[0], parameter::camera_trans_lag[1], parameter::camera_trans_lag[2]);
    glRotatef(parameter::camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(parameter::camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, parameter::modelView);
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        parameter::buttonState |= 1<<button;
    else if (state == GLUT_UP)
        parameter::buttonState = 0;

    int mods = glutGetModifiers();
    
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        parameter::buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        parameter::buttonState = 3;
    }

    parameter::ox = x;
    parameter::oy = y;

    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    cudaGLSetGLDevice(0);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_RGB | GLUT_DOUBLE); 
	glutInitWindowPosition(100,100);
	glutInitWindowSize(parameter::width, parameter::height); // Window Size If We Start In Windowed Mode
	glutCreateWindow("Symulacja skrzydla");
	init();
	
    // register callbacks
	glutDisplayFunc(renderScene); //display
	glutReshapeFunc(changeSize); //reshape
    //glutIdleFunc(idleFunction);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
	glutKeyboardFunc(keyFunction);
	
    // enter GLUT event processing cycle
	glutMainLoop();
    cudaGraphicsUnregisterResource(pointsVBO_Resource);
    glDeleteBuffers(1, &pointsVBO);
}
