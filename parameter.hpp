#ifndef PARAMETER_HPP
#define PARAMETER_HPP

namespace parameter
{
    const int width = 1200;
    const int height = 800;
    const int a = 100;
    
    const int maxParticles = 32000; 

    bool fullscreen = false;
    bool punkty = true; //tymczasowe rozwiazanie do wywalenia pozniej

    //do menu kontekstowego
    enum { CUDA, EXIT, reset, wspol, pudelko, skrzydelko, punkty_kulki, sila };	
    bool g_keys[256]; // Keys Array
    
    //parametry widoku - kamery
    int ox, oy;
    int buttonState = 0;
    float camera_trans[] = {0, 0, -3};
    float camera_rot[] = {0, 0, 0};
    float camera_trans_lag[] = {0, 0, -3};
    float camera_rot_lag[] = {0, 0, 0};
    const float inertia = 0.1;
    float modelView[16];

    static float colors[12][3] = {     // tecza kolorow
        {1.0f,0.5f,0.5f},{1.0f,0.75f,0.5f},{1.0f,1.0f,0.5f},{0.75f,1.0f,0.5f},
        {0.5f,1.0f,0.5f},{0.5f,1.0f,0.75f},{0.5f,1.0f,1.0f},{0.5f,0.75f,1.0f},
        {0.5f,0.5f,1.0f},{0.75f,0.5f,1.0f},{1.0f,0.5f,1.0f},{1.0f,0.5f,0.75f}
    };
} // namespace constant

#endif // PARAMETER_HPP
