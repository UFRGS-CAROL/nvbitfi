#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "real.h"

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;
struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    real ** sums;
    real * rand;
    real * cost;
    real * state;
    real * prev_state;
    real * forgot_state;
    real * forgot_delta;
    real * state_delta;
    real * combine_cpu;
    real * combine_delta_cpu;

    real * concat;
    real * concat_delta;

    real * binary_weights;

    real * biases;
    real * bias_updates;

    real * scales;
    real * scale_updates;

    real * weights;
    real * weight_updates;

    real * delta;
    real * output;
    real * loss;
    real * squared;
    real * norms;

    real * spatial_mean;
    real * mean;
    real * variance;

    real * mean_delta;
    real * variance_delta;

    real * rolling_mean;
    real * rolling_variance;

    real * x;
    real * x_norm;

    real * m;
    real * v;
    
    real * bias_m;
    real * bias_v;
    real * scale_m;
    real * scale_v;


    real *z_cpu;
    real *r_cpu;
    real *h_cpu;
    real * prev_state_cpu;

    real *temp_cpu;
    real *temp2_cpu;
    real *temp3_cpu;

    real *dh_cpu;
    real *hh_cpu;
    real *prev_cell_cpu;
    real *cell_cpu;
    real *f_cpu;
    real *i_cpu;
    real *g_cpu;
    real *o_cpu;
    real *c_cpu;
    real *dc_cpu; 

    real * binary_input;

/* --- Mixed precision --- */
    int real_type;

// #if MIX_PRECISION_SUPPORT == HALF
    // > Layers: Max Pool, Convolutional
    half_host *output_half;
    half_host *delta_half;
    // > Convolutional Layer
    half_host *weights_half;
    half_host *weight_updates_half;
    half_host *biases_half;
    half_host *bias_updates_half;
    half_host *binary_weights_half;
    half_host *scales_half;
    half_host *scale_updates_half;
    half_host *binary_input_half;
    half_host *mean_half;
    half_host *mean_delta_half;
    half_host *variance_half;
    half_host *variance_delta_half;
    half_host *rolling_mean_half;
    half_host *rolling_variance_half;
    half_host *x_half;
    half_host *x_norm_half;
    half_host *m_half;
    half_host *v_half;
    half_host *bias_m_half;
    half_host *scale_m_half;
    half_host *bias_v_half;
    half_host *scale_v_half;

    #ifdef GPU
        // > Layers: Max Pool, Convolutional
        half_host *delta_half_gpu;
        half_host *output_half_gpu;
        // > Convolutional Layer
        half_host *m_half_gpu;
        half_host *v_half_gpu;
        half_host *bias_m_half_gpu;
        half_host *bias_v_half_gpu;
        half_host *scale_m_half_gpu;
        half_host *scale_v_half_gpu;
        half_host *weights_half_gpu;
        half_host *weight_updates_half_gpu;
        half_host *biases_half_gpu;
        half_host *bias_updates_half_gpu;
        half_host *binary_weights_half_gpu;
        half_host *binary_input_half_gpu;
        half_host *mean_half_gpu;
        half_host *variance_half_gpu;
        half_host *rolling_mean_half_gpu;
        half_host *rolling_variance_half_gpu;
        half_host *mean_delta_half_gpu;
        half_host *variance_delta_half_gpu;
        half_host *scales_half_gpu;
        half_host *scale_updates_half_gpu;
        half_host *x_half_gpu;
        half_host *x_norm_half_gpu;
    #endif

// #elif MIX_PRECISION_SUPPORT == FLOAT
    // > Layers: Max Pool, Convolutional
    float *output_float;
    float *delta_float;
    // > Convolutional Layer
    float *weights_float;
    float *weight_updates_float;
    float *biases_float;
    float *bias_updates_float;
    float *binary_weights_float;
    float *scales_float;
    float *scale_updates_float;
    float *binary_input_float;
    float *mean_float;
    float *mean_delta_float;
    float *variance_float;
    float *variance_delta_float;
    float *rolling_mean_float;
    float *rolling_variance_float;
    float *x_float;
    float *x_norm_float;
    float *m_float;
    float *v_float;
    float *bias_m_float;
    float *scale_m_float;
    float *bias_v_float;
    float *scale_v_float;

    #ifdef GPU
        // > Layers: Max Pool, Convolutional
        float *delta_float_gpu;
        float *output_float_gpu;
        // > Convolutional Layer
        float *m_float_gpu;
        float *v_float_gpu;
        float *bias_m_float_gpu;
        float *bias_v_float_gpu;
        float *scale_m_float_gpu;
        float *scale_v_float_gpu;
        float *weights_float_gpu;
        float *weight_updates_float_gpu;
        float *biases_float_gpu;
        float *bias_updates_float_gpu;
        float *binary_weights_float_gpu;
        float *binary_input_float_gpu;
        float *mean_float_gpu;
        float *variance_float_gpu;
        float *rolling_mean_float_gpu;
        float *rolling_variance_float_gpu;
        float *mean_delta_float_gpu;
        float *variance_delta_float_gpu;
        float *scales_float_gpu;
        float *scale_updates_float_gpu;
        float *x_float_gpu;
        float *x_norm_float_gpu;
    #endif
// #endif
/* ---------- x ---------- */

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

#ifdef GPU
    int *indexes_gpu;

    real *z_gpu;
    real *r_gpu;
    real *h_gpu;

    real *temp_gpu;
    real *temp2_gpu;
    real *temp3_gpu;

    real *dh_gpu;
    real *hh_gpu;
    real *prev_cell_gpu;
    real *cell_gpu;
    real *f_gpu;
    real *i_gpu;
    real *g_gpu;
    real *o_gpu;
    real *c_gpu;
    real *dc_gpu; 

    real *m_gpu;
    real *v_gpu;
    real *bias_m_gpu;
    real *scale_m_gpu;
    real *bias_v_gpu;
    real *scale_v_gpu;

    real * combine_gpu;
    real * combine_delta_gpu;

    real * prev_state_gpu;
    real * forgot_state_gpu;
    real * forgot_delta_gpu;
    real * state_gpu;
    real * state_delta_gpu;
    real * gate_gpu;
    real * gate_delta_gpu;
    real * save_gpu;
    real * save_delta_gpu;
    real * concat_gpu;
    real * concat_delta_gpu;

    real * binary_input_gpu;
    real * binary_weights_gpu;

    real * mean_gpu;
    real * variance_gpu;

    real * rolling_mean_gpu;
    real * rolling_variance_gpu;

    real * variance_delta_gpu;
    real * mean_delta_gpu;

    real * x_gpu;
    real * x_norm_gpu;
    real * weights_gpu;
    real * weight_updates_gpu;
    real * weight_change_gpu;

    real * biases_gpu;
    real * bias_updates_gpu;
    real * bias_change_gpu;

    real * scales_gpu;
    real * scale_updates_gpu;
    real * scale_change_gpu;

    real * output_gpu;
    real * loss_gpu;
    real * delta_gpu;
    real * rand_gpu;
    real * squared_gpu;
    real * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;
typedef struct network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    real *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    real *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    real *input;
    real *truth;
    real *delta;
    real *workspace;
    int train;
    int index;
    real *cost;
    float clip;

    int input_data_type;

#ifdef GPU
    real *input_gpu;
    real *truth_gpu;
    real *delta_gpu;
    real *output_gpu;

    real *hold_input_gpu;
#endif

/* --- Mixed precision --- */
// #if MIX_PRECISION_SUPPORT == HALF
    half_host *input_half;
    half_host *workspace_half;
    
    #ifdef GPU
        half_host *input_half_gpu;
        half_host *hold_input_half_gpu;
    #endif
// #elif MIX_PRECISION_SUPPORT == FLOAT
    float *input_float;
    float *workspace_float;
    
    #ifdef GPU
        float *input_float_gpu;
        float *hold_input_float_gpu;
    #endif
// #endif
/* ---------- x ---------- */
} network;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    real *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;


typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;


network *load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

// > Headers

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);

void softmax(real *input, int n, float temp, int stride, real *output);
int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void axpy_gpu(int N, float ALPHA, double *X, int INCX, double *Y, int INCY);
void axpy_gpu(int N, float ALPHA, half_host *X, int INCX, half_host *Y, int INCY);

void fill_gpu(int N, float ALPHA, float * X, int INCX);
void fill_gpu(int N, float ALPHA, double * X, int INCX);
void fill_gpu(int N, float ALPHA, half_host * X, int INCX);

void scal_gpu(int N, float ALPHA, float *X, int INCX);
void scal_gpu(int N, float ALPHA, double *X, int INCX);
void scal_gpu(int N, float ALPHA, half_host *X, int INCX);

void copy_gpu(int N, float *X, int INCX, float *Y, int INCY);
void copy_gpu(int N, double *X, int INCX, double *Y, int INCY);
void copy_gpu(int N, half_host *X, int INCX, half_host *Y, int INCY);


void cuda_set_device(int n);
real *cuda_make_array(real *x, size_t n);
float *cuda_make_float_array(float *x, size_t n);
half_host *cuda_make_half_array(half_host *x, size_t n);

float cuda_mag_array(real *x_gpu, size_t n);
void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

float train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, real *X, real *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
real *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);


void fuse_conv_batchnorm(network net);
void replace_image_to_label(char *input_path, char *output_path);
network *parse_network_cfg_custom(char *filename, int batch);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image real_to_image(int w, int h, int c, real *data);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, float rad);
void visualize_network(network *net);
float box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box real_to_box(real *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);

real *network_predict(network *net, real *input);

#if REAL != FLOAT
real *network_predict(network *net, float *input_float);
#endif

int network_width(network *net);
int network_height(network *net);
real *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(image m);
float train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(real *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
int *read_map(char *filename);
void error(const char *s);
int max_int_index(int *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(real *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

// > Templates

#include "blas_templates.h"
#include "utils_templates.h"
#include "cuda_templates.h"
#include "activation_templates.h"

#endif
