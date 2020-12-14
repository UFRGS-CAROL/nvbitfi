#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

// extern "C" {
#include "crop_layer.h"
#include "utils.h"
#include "cuda.h"
#include "image.h"
// }

/* make_real3
 * Based on CUDA vector_functions.h
 */
 __device__ real_device3 make_real3(real_device x, real_device y, real_device z) {
    real_device3 t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}

__device__ real_device get_pixel_kernel(real_device *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

__device__ real_device3 rgb_to_hsv_kernel(real_device3 rgb)
{
    real_device r = rgb.x;
    real_device g = rgb.y; 
    real_device b = rgb.z;

    real_device h, s, v;
    real_device max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    real_device min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    real_device delta = max - min;
    v = max;
    if(max == CAST_DEV(0)){
        s = 0;
        h = -1;
    }else{
        s = delta/max;
        if(r == max){
            h = (g - b) / delta;
        } else if (g == max) {
            h = CAST_DEV(2) + (b - r) / delta;
        } else {
            h = CAST_DEV(4) + (r - g) / delta;
        }
        if (h < CAST_DEV(0)) h += 6;
    }
    return make_real3(h, s, v);
}

__device__ real_device3 hsv_to_rgb_kernel(real_device3 hsv)
{
    real_device h = hsv.x;
    real_device s = hsv.y; 
    real_device v = hsv.z;

    real_device r, g, b;
    real_device f, p, q, t;

    if (s == CAST_DEV(0)) {
        r = g = b = v;
    } else {
        int index = (int) floor_real(h);
        f = h - CAST_DEV(index);
        p = v*(CAST_DEV(1)-s);
        q = v*(CAST_DEV(1)-s*f);
        t = v*(CAST_DEV(1)-s*(CAST_DEV(1)-f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < CAST_DEV(0)) ? CAST_DEV(0) : ((r > CAST_DEV(1)) ? CAST_DEV(1) : r);
    g = (g < CAST_DEV(0)) ? CAST_DEV(0) : ((g > CAST_DEV(1)) ? CAST_DEV(1) : g);
    b = (b < CAST_DEV(0)) ? CAST_DEV(0) : ((b > CAST_DEV(1)) ? CAST_DEV(1) : b);
    return make_real3(r, g, b);
}

__device__ real_device bilinear_interpolate_kernel(real_device *image, int w, int h, real_device x, real_device y, int c)
{
    int ix = (int) floor_real(x);
    int iy = (int) floor_real(y);

    real_device dx = x - CAST_DEV(ix);
    real_device dy = y - CAST_DEV(iy);

    real_device val = (CAST_DEV(1)-dy) * (CAST_DEV(1)-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + 
        dy     * (CAST_DEV(1)-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + 
        (CAST_DEV(1)-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

__global__ void levels_image_kernel(real_device *image, real_device *rand, int batch, int w, int h, int train, real_device saturation, real_device exposure, real_device translate, real_device scale, real_device shift)
{
    int size = batch * w * h;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    real_device rshift = rand[0];
    real_device gshift = rand[1];
    real_device bshift = rand[2];
    real_device r0 = rand[8*id + 0];
    real_device r1 = rand[8*id + 1];
    real_device r2 = rand[8*id + 2];
    real_device r3 = rand[8*id + 3];

    saturation = r0*(saturation - CAST_DEV(1)) + CAST_DEV(1);
    saturation = (r1 > CAST_DEV(.5f)) ? CAST_DEV(1.f)/saturation : saturation;
    exposure = r2*(exposure - CAST_DEV(1)) + CAST_DEV(1);
    exposure = (r3 > CAST_DEV(.5f)) ? CAST_DEV(1.f)/exposure : exposure;

    size_t offset = id * h * w * 3;
    image += offset;
    real_device r = image[x + w*(y + h*0)];
    real_device g = image[x + w*(y + h*1)];
    real_device b = image[x + w*(y + h*2)];

    real_device3 rgb = make_real3(r,g,b);

    if(train){
        real_device3 hsv = rgb_to_hsv_kernel(rgb);
        hsv.y *= saturation;
        hsv.z *= exposure;
        rgb = hsv_to_rgb_kernel(hsv);
    } else {
        shift = 0;
    }
    image[x + w*(y + h*0)] = rgb.x*scale + translate + (rshift - CAST_DEV(.5f))*shift;
    image[x + w*(y + h*1)] = rgb.y*scale + translate + (gshift - CAST_DEV(.5f))*shift;
    image[x + w*(y + h*2)] = rgb.z*scale + translate + (bshift - CAST_DEV(.5f))*shift;
}

__global__ void forward_crop_layer_kernel(real_device *input, real_device *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, real_device angle, real_device *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    real_device cx = w/2.f;
    real_device cy = h/2.f;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    real_device r4 = rand[8*b + 4];
    real_device r5 = rand[8*b + 5];
    real_device r6 = rand[8*b + 6];
    real_device r7 = rand[8*b + 7];

    real_device dw = CAST_DEV(w - crop_width)*r4;
    real_device dh = CAST_DEV(h - crop_height)*r5;
    flip = (flip && (r6 > CAST_DEV(.5f)));
    angle = CAST_DEV(2)*angle*r7 - angle;
    if(!train){
        dw = (w - crop_width)/2.f;
        dh = (h - crop_height)/2.f;
        flip = 0;
        angle = 0;
    }

    input += w*h*c*b;

    real_device x = (flip) ? CAST_DEV(w) - dw - CAST_DEV(j - 1) : CAST_DEV(j) + dw;    
    real_device y = CAST_DEV(i) + dh;

    real_device rx = cos_real(angle)*(x-cx) - sin_real(angle)*(y-cy) + cx;
    real_device ry = sin_real(angle)*(x-cx) + cos_real(angle)*(y-cy) + cy;

    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}

void forward_crop_layer_gpu(crop_layer layer, network net)
{
    cuda_random(layer.rand_gpu, layer.batch*8);

    float radians = layer.angle*(3.14159265f/180.f);

    float scale = 2;
    float translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }

    int size = layer.batch * layer.w * layer.h;

    levels_image_kernel<<<cuda_gridsize(size), BLOCK>>>((real_device*)net.input_gpu, (real_device*)layer.rand_gpu, layer.batch, layer.w, layer.h, net.train, (real_device)layer.saturation, (real_device)layer.exposure, (real_device)translate, (real_device)scale, (real_device)layer.shift);
    check_error(cudaPeekAtLastError());

    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>((real_device*)net.input_gpu, (real_device*)layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, net.train, layer.flip, (real_device)radians, (real_device*)layer.output_gpu);
    check_error(cudaPeekAtLastError());
}
