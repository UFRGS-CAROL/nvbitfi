#ifndef COL2IM_H
#define COL2IM_H

#include "darknet.h"

void col2im_cpu(real* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, real* data_im);

#ifdef GPU
void col2im_gpu(real *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, real *data_im);
#endif
#endif
