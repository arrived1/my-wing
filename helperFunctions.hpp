#ifndef HRLPER_FUNCTION_HPP
#define HELPER_FUNCTION_HPP

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>

#include <parameter.hpp>
#include <wing.hpp>

extern Parameter parameter;
extern Wing wing;
extern float3 points[N];
extern float3 velocities[N];

void initializePositionsAndVelocities();
GLuint compileProgram(const char *vsource, const char *fsource);
void renderWing();
void renderBox(int boxSize);
void changeSize(int w, int h); 
void keyFunction(unsigned char key, int, int); 
void motion(int x, int y);
void camera();
void mouse(int button, int state, int x, int y);

#endif // HELPER_FUNCTION_HPP
