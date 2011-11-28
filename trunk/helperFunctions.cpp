#include <helperFunctions.hpp>

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

void renderBox(int boxSize)
{
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(boxSize);

    glColor3f(1.0, .0, .0);
    glBegin(GL_LINES);
        glVertex3f(.0f, .0f, .0f);  //x
        glVertex3f(boxSize/2 + 20.0f, .0f, .0f);

        glVertex3f(boxSize/2 + 20.0f, .0f, .0f);	//strzalka gora
        glVertex3f(boxSize/2 + 15.0f, 2.0f, .0f);

        glVertex3f(boxSize/2 + 20.0f, .0f, .0f);	//strzalka dol
        glVertex3f(boxSize/2 + 15.0f, -2.0f, .0f);

        glVertex3f(.0f, .0f, .0f);  //y
        glVertex3f(.0f, boxSize/2 + 20.0f, .0f);

        glVertex3f(.0f, boxSize/2 + 20.0f, .0f);	//strzalka prawo
        glVertex3f(2.0f, boxSize/2 + 15.0f, .0f);

        glVertex3f(.0f, boxSize/2 + 20.0f, .0f);	//strzalka lewo
        glVertex3f(-2.0f, boxSize/2 + 15.0f, .0f);

        glVertex3f(.0f, .0f, .0f);  //z
        glVertex3f(.0f, .0f, boxSize/2 + 20.0f);

        glVertex3f(.0f, .0f, boxSize/2 + 20.0f);	//strzalka lewo
        glVertex3f(-2.0f, .0f, boxSize/2 + 15.0f);

        glVertex3f(.0f, .0f, boxSize/2 + 20.0f);	//strzalka prawo
        glVertex3f(2.0f, .0f, boxSize/2 + 15.0f);
    glEnd();
}

void changeSize(int w, int h) 
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(90.0, (float) w / (float) h, 0.1, parameter.a + 400.0);
    glTranslatef(0, 0, -parameter.a);
   
    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);
}

void keyFunction(unsigned char key, int, int) 
{
	if (key == 27)
        exit(0);
}

void motion(int x, int y)
{
    float dx = x - parameter.ox;
    float dy = y - parameter.oy;

    if (parameter.buttonState == 3) 
    {
        // left+middle = zoom
        parameter.camera_trans[2] += (dy / 100.0) * 0.5 * fabs(parameter.camera_trans[2]);
    } 
    else if (parameter.buttonState & 2) 
    {
        // middle = translate
        parameter.camera_trans[0] += dx / 10.0;
        parameter.camera_trans[1] -= dy / 10.0;
    }
    else if (parameter.buttonState & 1) 
    {
        // left = rotate
        parameter.camera_rot[0] += dy / 5.0;
        parameter.camera_rot[1] += dx / 5.0;
    }

    parameter.ox = x; 
    parameter.oy = y;

    glutPostRedisplay();
}

void camera()
{
	// view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    for (int c = 0; c < 3; ++c)
    {
        parameter.camera_trans_lag[c] += 
            (parameter.camera_trans[c] - parameter.camera_trans_lag[c]) * parameter.inertia;
        parameter.camera_rot_lag[c] += 
            (parameter.camera_rot[c] - parameter.camera_rot_lag[c]) * parameter.inertia;
    }
    glTranslatef(parameter.camera_trans_lag[0], parameter.camera_trans_lag[1], parameter.camera_trans_lag[2]);
    glRotatef(parameter.camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(parameter.camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, parameter.modelView);
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        parameter.buttonState |= 1<<button;
    else if (state == GLUT_UP)
        parameter.buttonState = 0;

    int mods = glutGetModifiers();
    
    if (mods & GLUT_ACTIVE_SHIFT) 
    {
        parameter.buttonState = 2;
    } 
    else if (mods & GLUT_ACTIVE_CTRL) 
    {
        parameter.buttonState = 3;
    }

    parameter.ox = x;
    parameter.oy = y;

    glutPostRedisplay();
}

