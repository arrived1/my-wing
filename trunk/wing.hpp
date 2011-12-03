#ifndef WING_HPP
#define WING_HPP

#include <cuda_runtime.h>

struct Wing
{
    Wing(float x = -10, float y = 0, float z = 0, float hight = 30)
        : pos(make_float3(x, y, z)), hight(hight), wingRadius(4), angle(0)
    {}

    float3 pos;
    float hight;
    float wingRadius;
    float angle;
};


#endif // WING_HPP
