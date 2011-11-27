// nvcc main.cpp kernels.cu -lglut –lGLU
#include <iostream>
#include <cmath>
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <GL/glut.h>

#include <kernels.h>
#include <parameter.hpp>
#include <helperFunctions.hpp>

GLuint pointsVBO;
float3 *pointsVBO_device; // device pointer for VBO
struct cudaGraphicsResource *pointsVBO_Resource;
float3 points[N];
float3 velocities[N];
Parameter parameter;

void initializePositionsAndVelocities()
{
    int counter = 0;
	for(float x = -49; x < -39; x++)	//1000
		for(float y = -5; y < 5; y++)
			for(float z = -5; z < 5; z++)
			{	
				points[counter] = make_float3(x, y, z);
                velocities[counter] = make_float3(10, 0, 0);
				counter++;
			}
}


void init(void)
{
	// GLEW library
	GLenum err = glewInit();
	if (GLEW_OK != err)		
        exit(0);
	
    initializePositionsAndVelocities();

    // copy velocity -> device
    void **v_device_addr;
    cudaGetSymbolAddress ((void **)&v_device_addr, "v_device");
    cudaMemcpy (v_device_addr, velocities, sizeof(float3) * N,  cudaMemcpyHostToDevice );
	
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



void renderScene1(void) 
{
    camera();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderBox(parameter.a);

	glColor4f(1,1,0,0.7);
	glPointSize(3.0);
	
    // draw vbo
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, NULL); //(number of coordinates, type, stride, pointer)
    //glColorPonter(_, _, _); //TODO: colors
	glDrawArrays(GL_POINTS, 0, N);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();
}

#define STRINGIFY(A) #A

// vertex shader
const char *vertexShader = STRINGIFY(
uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels
uniform float densityScale;
uniform float densityOffset;
void main()
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    gl_FrontColor = gl_Color;
}
);

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader = STRINGIFY(
void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

    gl_FragColor = gl_Color * diffuse;
}
);

GLuint compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);
    
    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void renderScene(void)
{
    camera();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderBox(parameter.a);
    
    GLuint m_program = compileProgram(vertexShader, spherePixelShader);
    float m_fov = 60;

    glEnable(GL_POINT_SPRITE_ARB);
    glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(m_program);
    glUniform1f( glGetUniformLocation(m_program, "pointScale"), parameter.height
            / tanf(m_fov*0.5f*(float)M_PI/180.0f) );
    glUniform1f( glGetUniformLocation(m_program, "pointRadius"), 0.1 );

    glColor3f(1, 1, 0);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, pointsVBO);

        glEnableClientState(GL_VERTEX_ARRAY);                
        glVertexPointer(3, GL_FLOAT, 0, 0);

        glDrawArrays(GL_POINTS, 0, N);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
        glDisableClientState(GL_VERTEX_ARRAY); 
        //glDisableClientState(GL_COLOR_ARRAY); 
    
    glUseProgram(0);
    glDisable(GL_POINT_SPRITE_ARB);

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

int main(int argc, char **argv)
{
    cudaGLSetGLDevice(0);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_RGB | GLUT_DOUBLE); 
	glutInitWindowPosition(100,100);
	glutInitWindowSize(parameter.width, parameter.height); // Window Size If We Start In Windowed Mode
	glutCreateWindow("Symulacja skrzydla");
	init();
	
    // register callbacks
	glutDisplayFunc(renderScene); //display
	glutReshapeFunc(changeSize); //reshape
    glutIdleFunc(idleFunction);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
	glutKeyboardFunc(keyFunction);
	
    // enter GLUT event processing cycle
	glutMainLoop();
    cudaGraphicsUnregisterResource(pointsVBO_Resource);
    glDeleteBuffers(1, &pointsVBO);
}
