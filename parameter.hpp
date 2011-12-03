#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#define N 1000//32000 //16000
#define box 100

struct Parameter
{
    Parameter()
        : width(1200),
        height(800),
        a(box),
        maxParticles(N),
        inertia(0.1)
    {
        fullscreen = false;
        buttonState = 0;
        camera_trans[0] = 0;
        camera_trans[1] = 0;
        camera_trans[2] = -3;
        camera_rot[0] = 0;
        camera_rot[1] = 0;
        camera_rot[2] = 0;
        camera_trans_lag[0] = 0;
        camera_trans_lag[1] = 0;
        camera_trans_lag[2] = -3;
        camera_rot_lag[0] = 0;
        camera_rot_lag[1] = 0;
        camera_rot_lag[2] = 0;
    }
        
    const int width;
    const int height;
    const int a;

    const int maxParticles;

    bool fullscreen;
    bool punkty;

    //do menu kontekstowego
    bool g_keys[256]; // Keys Array

    //parametry widoku - kamery
    int ox, oy;
    int buttonState;
    const float inertia;
    float camera_trans[3];
    float camera_rot[3];
    float camera_trans_lag[3];
    float camera_rot_lag[3];
    float modelView[16];
    //static float colors[12][3] = {     // tecza kolorow
        //{1.0f,0.5f,0.5f},{1.0f,0.75f,0.5f},{1.0f,1.0f,0.5f},{0.75f,1.0f,0.5f},
        //{0.5f,1.0f,0.5f},{0.5f,1.0f,0.75f},{0.5f,1.0f,1.0f},{0.5f,0.75f,1.0f},
        //{0.5f,0.5f,1.0f},{0.75f,0.5f,1.0f},{1.0f,0.5f,1.0f},{1.0f,0.5f,0.75f}
    //};
};

#endif // PARAMETER_HPP
