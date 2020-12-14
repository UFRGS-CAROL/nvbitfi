#include "darknet.h"
#include "blas.h"
#include <libgen.h>

// > Auxiliary structs

#define LAYERS_OUT_COMP_INTVL_LIMITS 3
typedef struct  {
    double err_avg;
    double err_max;
    int err_max_index;
    double err_min;
    float intvl_limits[LAYERS_OUT_COMP_INTVL_LIMITS];
    int intvl_counts[LAYERS_OUT_COMP_INTVL_LIMITS+1];
} layersOutComp;

typedef struct {
    float avgArea;
    float stdevArea;
    float minArea;
    float maxArea;
    int numObjs;
} LabelsMetrics;

// > Auxiliary functions

void print_detections(detection *dets, int num, float thresh, char **names, int classes) {
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int _class = -1;
        for(j = 0; j < classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                if (_class < 0) {
                    strcat(labelstr, names[j]);
                    _class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
    }
}

/* Save array to file
 * Binary file format
 *     - n               (int   - 32 bits)
 *     - output array    (float - n*32 bits)
 */
template <typename T>
void save_array_to_file(T *output, int n, char *filename, int bin) {
    FILE *f;
    int i;

    if (bin) {
        f = fopen(filename, "wb");
        float *output_float;

        if (typeid(T) == typeid(float))
            output_float = (float*)output;
        else if (typeid(T) == typeid(half_host)) {
            output_float = (float*)calloc(n, sizeof(float));
            for (i = 0; i < n; i++)
                output_float[i] = output[i];
        }

        fwrite(&n, sizeof(int), 1, f);
        fwrite(output_float, sizeof(float), n, f);

        if (typeid(T) == typeid(half_host)) free(output_float);
    } else {
        f = fopen(filename, "w");
        for (i = 0; i < n; i++)
            fprintf(f, "%f\n", (float)output[i]);
    }

    fclose(f);
}

/* Save layer output to file
 * - bin [int 0|1]: whether output file will be binary or text
 * [Binary file format]
 *    -> n               (int   - 32 bits)
 *    -> output array    (float - n*32 bits)
 */
void save_layer_output_to_file(layer l, char *outputFile, int bin) {
    int outputs = l.outputs * l.batch;
    if (IS_MIX_PRECISION_FLOAT_LAYER(l.real_type)) {
        cuda_pull_array(l.output_float_gpu, l.output_float, outputs);
        save_array_to_file(l.output_float, outputs, outputFile, bin);
    } else if (IS_MIX_PRECISION_HALF_LAYER(l.real_type)) {
        cuda_pull_array(l.output_half_gpu, l.output_half, outputs);
        save_array_to_file(l.output_half, outputs, outputFile, bin);
    } else {
        cuda_pull_array(l.output_gpu, l.output, outputs);
        save_array_to_file(l.output, outputs, outputFile, bin);
    }
}

layersOutComp *compare_layers_output(float *floatOut, half_host *halfOut, int n) {
    layersOutComp *res = (layersOutComp*)calloc(1, sizeof(layersOutComp));

    double max = 0; int max_index = -1;
    double min = 100;
    double sum = 0;

    float intvl_limits[LAYERS_OUT_COMP_INTVL_LIMITS] = {1, 10, 100};
    int intvl_counts[LAYERS_OUT_COMP_INTVL_LIMITS+1] = {0, 0, 0, 0};

    int i, j;
    for (i = 0; i < n; i++) {
        if (abs(floatOut[i]) == 0) continue;
        double rel_err = abs(floatOut[i] - halfOut[i])/abs(floatOut[i]);

        sum += rel_err;
        if (max < rel_err) { max = rel_err; max_index = i; }
        if (min > rel_err) min = rel_err;

        for (j = 0; j < LAYERS_OUT_COMP_INTVL_LIMITS+1; j++) {
            float inf = (j-1 < 0) ? 0.0 : intvl_limits[j-1];
            float sup = (j >= LAYERS_OUT_COMP_INTVL_LIMITS) ? INFINITY : intvl_limits[j]; 
            if (rel_err > inf && rel_err <= sup) intvl_counts[j]++;
        }
    }

    res->err_avg = sum/n;
    res->err_max = max;
    res->err_max_index = max_index;
    res->err_min = min;
    memcpy(res->intvl_limits, intvl_limits, LAYERS_OUT_COMP_INTVL_LIMITS*sizeof(float));
    memcpy(res->intvl_counts, intvl_counts, (LAYERS_OUT_COMP_INTVL_LIMITS+1)*sizeof(int));

    return res;
}

double array_average(float *array, int n) {
    double sum = 0;
    int i;
    for (i = 0; i < n; i++) sum += array[i];
    return sum/n;
}

LabelsMetrics calcLabelsMetrics(box_label *labels, int n) {
    LabelsMetrics m;
    int i;
    
    float minArea = MAXFLOAT;
    float maxArea = -MAXFLOAT;
    float avgArea = 0;
    float stdevArea = 0;
    for (i = 0; i < n; i++) {
        float A = labels[i].w * labels[i].h;
        if (A < minArea) minArea = A;
        if (A > maxArea) maxArea = A;
        avgArea += A;
    }
    avgArea /= n;

    for (i = 0; i < n; i++) {
        float A = labels[i].w * labels[i].h;
        stdevArea += pow(A - avgArea, 2);
    }
    stdevArea = sqrt(stdevArea/n);

    m.avgArea = avgArea;
    m.stdevArea = stdevArea;
    m.minArea = minArea;
    m.maxArea = maxArea;
    m.numObjs = n;

    return m;
}

// > Tests

/* Test 1
 * Description: Test average execution time for one frame after 'n' predictions (default n=1)
 * Call: ./darknet detector rtest 1 <cfgfile> <weightfile> <filename> [-n <iterations>] [-print <0|1>]
 * Optionals: -n <iterations> => default: 1
 *            -print <0|1>    => print detections? default: 1
 */
void test1(char *cfgfile, char *weightfile, char *filename, int n, int print) {
    char *datacfg = (char*)"cfg/coco.data";
    float thresh = 0.3;
    float hier_thresh = 0.5;

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, (char*)"names", (char *)"data/names.list");
    char **names = get_labels(name_list);

    double tl = what_time_is_it_now();

    network *net = load_network(cfgfile, weightfile, 0);

    printf("Load net time: %f ms.\n\n", (what_time_is_it_now() - tl) * 1000);

    set_batch_network(net, 1);
    srand(2222222);

    char buff[256];
    char *input = buff;
    float nms = .45;

    strncpy(input, filename, 255);

    layer l = net->layers[net->n - 1];

    image im, sized;
    int nboxes = 0;
    detection *dets;

    double t = what_time_is_it_now();

    int i;
    for (i = 0; i < n; i++) {
        // Load input image
        im = load_image_color(input, 0, 0);
        sized = letterbox_image(im, net->w, net->h);

        float *X = sized.data;

        // Run predictor
        network_predict(net, X);

        // Generate outputs
        int letterbox = 1;
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);

        if (nms)
            do_nms_sort(dets, nboxes, l.classes, nms);

        if (print) {
            print_detections(dets, nboxes, thresh, names, l.classes);
            printf("\n");
        } else {
            fprintf(stderr, "\r%d ", i+1);
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
    }

    double tt = (what_time_is_it_now() - t) * 1000;
    printf("\nTotal time: %f ms.\n", tt);
    if (n > 1)
        printf("Average time: %f ms.\n", tt/n);

    free_network(net);
}

/* Test 2
 * Description: Test average execution time for one frame with 'n' COCO validation images
 * Details: First prediction time is discarded because of network push cost
 * Call: ./darknet detector rtest 2 <cfgfile> <weightfile> [-n <images>]
 * Optionals: -n <images> - number of images from COCO dataset (max: 4999, min: 2, default: 5000)
 */
void test2(char *cfgfile, char *weightfile, int n) {
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(22222);
    layer l = net->layers[net->n-1];

    char *valid_images = (char*)"../coco_test/5k.txt";
    list *plist = get_paths(valid_images);
    char **paths = (char**)list_to_array(plist);
    if (n > plist->size) {
        fprintf(stderr, "Argument 'n' cannot be greater than %d!", plist->size);
        return;
    }

    image im, sized;
    int nboxes = 0;
    detection *dets;

    float nms = .45;
    float thresh = 0.3;
    float hier_thresh = 0.5;

    double t1 = 0;

    int i;
    for (i = 0; i < n; i++) {
        if (i == 1) // Discard first frame (because of network push cost)
            t1 = what_time_is_it_now();

        im = load_image_color(paths[i], 0, 0);
        sized = letterbox_image(im, net->w, net->h);
        float *X = sized.data;

        network_predict(net, X);

        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, 1);
        if (nms)
            do_nms_sort(dets, nboxes, l.classes, nms);
        // print_detections(dets, nboxes, thresh, names, l.classes);

        free_image(im);
        free_image(sized);
        free_detections(dets, nboxes);

        fprintf(stderr, "\r%d ", i+1);
    }

    double t = what_time_is_it_now() - t1;
    printf("\nTotal time for %d frame(s): %f s\n", n, t);
    printf("Time per frame: %f ms\n", t/n * 1000);

    free_network(net);
}

/* Test 3
 * Description: Test average execution time (in ms) for each layer after 'n' predictions
 * Call: ./darknet detector rtest 3 <cfgfile> <weightfile> <filename> [-n <iterations>]
 * Details: First prediction time is discarded because of network push cost
 *          IMPORTANT: set LAYERS_TIME_TEST to 1 in network.c!
 * Optionals: -n <iterations> => default: 50
 */

/* Format: layers_times[layer][iteration] */
float **layers_times;

void test3(char *cfgfile, char *weightfile, char *filename, int n) {
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char buff[256];
    char *input = buff;
    strncpy(input, filename, 255);

    image im = load_image_color(input, 0, 0);
    image sized = letterbox_image(im, net->w, net->h);
    float *X = sized.data;

    // Alloc times matrix
    int j;
    layers_times = (float**)malloc(net->n * sizeof(float*));
    for (j = 0; j < net->n; j++)
        layers_times[j] = (float*)malloc(n * sizeof(float));

    int i;
    for (i = 0; i < n; i++) {
        network_predict(net, X);
        fprintf(stderr, "\r%d ", i+1);
    }

    // Calculate average (discarding first iteration of each layer)
    printf("\n");
    for (i = 0; i < net->n; i++) {
        double sum = 0;
        for (j = 1; j < n; j++) sum += layers_times[i][j];
        double avg = sum/(n-1);
        printf("%f\n", avg*1000);
    }

    free_image(im);
    free_image(sized);
    free_network(net);    
}

/* Test 4
 * Description: Save output of the layer indicated by 'layerIndex' (related to 'Test 5')
 * Call: ./darknet detector rtest 4 <cfgfile> <weightfile> <inputFile> <layerIndex> <outputFile> [-bin <0|1>]
 * Optionals: -bin <0|1> => Output file type (0: txt, 1: binary), default: 1 (binary)
 */
void test4(char *cfgfile, char *weightfile, char *inputFile, int layerIndex, char *outputFile, int bin) {
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    char buff[256];
    char *input = buff;
    strncpy(input, inputFile, 255);

    image im = load_image_color(input, 0, 0);
    image sized = letterbox_image(im, net->w, net->h);
    float *X = sized.data;

    network_predict(net, X);

    layer l = net->layers[layerIndex];

    save_layer_output_to_file(l, outputFile, bin);

    printf("Output of layer %d saved to file: %s\n", layerIndex, outputFile);

    free_image(im);
    free_image(sized);
    free_network(net);    
}

/* Test 5
 * Description: Compare output of layers (related to 'Test 4')
 * Call: ./darknet detector rtest 5 <inputFile1> <inputFile2>
 * Details: input files must be binary files
 */
void test5(char *inputFile1, char *inputFile2) {
    FILE *f1 = fopen(inputFile1, "rb");
    FILE *f2 = fopen(inputFile2, "rb");

    if (f1 == NULL) {
        printf("Could not open file: %s\n", inputFile1);
        return;
    }

    if (f2 == NULL) {
        printf("Could not open file: %s\n", inputFile2);
        return;
    }

    int n1, n2;

    fread(&n1, sizeof(int), 1, f1);
    fread(&n2, sizeof(int), 1, f2);

    if (n1 != n2) {
        printf("Invalid input files: sizes doesn't match!\n");
        fclose(f1);
        fclose(f2);
        return;
    }

    int n = n1;
    int i;

    float *arr1 = (float*)calloc(n, sizeof(float));
    float *arr2 = (float*)calloc(n, sizeof(float));

    fread(arr1, sizeof(float), n, f1);
    fread(arr2, sizeof(float), n, f2);

    double max = 0;
    int max_index = -1;
    double min = 100;
    double sum = 0;

    // High error
    int high_n = 10;
    int high_count = 0;
    int high_index[high_n];
    float thresh = 20;

    for (i = 0; i < n; i++) {
        double rel_err = abs(arr1[i] - arr2[i])/abs(arr1[i]);
        sum += rel_err;
        if (max < rel_err) { max = rel_err; max_index = i; }
        if (min > rel_err) min = rel_err;

        if (rel_err >= thresh) {
            if (high_count < high_n) high_index[high_count] = i;
            high_count++;
        }
    }

    printf("===== Comparison results =====\n");
    printf("> Files: %s X %s\n", inputFile1, inputFile2);
    printf("> Average error: %f\n", sum/n);
    printf("> Max error: %f (%f X %f)\n", max, arr1[max_index], arr2[max_index]);
    printf("> Min error: %f\n", min);

    printf("> High error count (thresh: %.1f%%): %d (%f%%)\n", thresh*100, high_count, (float)high_count/n*100);
    for (i = 0; i < ((high_count > high_n) ? high_n : high_count); i++) {
        int index = high_index[i];
        printf(" - %7d: %+f X %+f (%.1f%%)\n", index, arr1[index], arr2[index], abs(arr1[index] - arr2[index])/abs(arr1[index])*100);
    }

    printf("==============================\n");

    free(arr1);
    free(arr2);
    fclose(f1);
    fclose(f2);
}

/* Test 6
 * Description: Compare (float VS. half) outputs of last two layers (105 and 106) of YOLOv3 on 'n' COCO validation images
 * Call: ./darknet detector rtest 6 [-n <images>]
 * Optionals: -n <images> - number of images from COCO dataset (max: 4999, min: 2, default: 4999)
 * IMPORTANT: REAL=float (on Makefile)
 */
void test6(int n) {
    if (REAL != FLOAT) {
        printf("Default REAL must be FLOAT (REAL=float on Makefile)!\n");
        return;
    }

    char *cfgfile_float = (char*)"cfg/yolov3.cfg";
    char *cfgfile_half = (char*)"cfg/yolov3-half.cfg";
    char *weightfile = (char*)"../yolov3.weights";

    network *netFloat = load_network(cfgfile_float, weightfile, 0);
    set_batch_network(netFloat, 1);
    network *netHalf = load_network(cfgfile_half, weightfile, 0);
    set_batch_network(netHalf, 1);
    srand(2222222);

    char *valid_images = (char*)"../coco_test/5k.txt";
    list *plist = get_paths(valid_images);
    char **paths = (char**)list_to_array(plist);
    image im, sized;
    if (n > plist->size) {
        fprintf(stderr, "Argument 'n' cannot be greater than %d!", plist->size);
        return;
    }

    char *datacfg = (char*)"cfg/coco.data";
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, (char*)"names", (char *)"data/names.list");
    char **class_names = get_labels(name_list);


    double t1 = what_time_is_it_now();

    layer l105_float = netFloat->layers[105], l105_half = netHalf->layers[105];
    layer l106_float = netFloat->layers[106], l106_half = netHalf->layers[106];
    int l105_outputs = l105_float.outputs;
    int l106_outputs = l106_float.outputs;
    layersOutComp *l105_cmp, *l106_cmp;

    char *img_name;
    char label_path[1000];
    int num_labels = 0;

    FILE *outFile = fopen("layer105x106-halfXfloat.csv", "w");
    if (!outFile) {
        perror("Could not create output file: ");
        return;
    }

    int i, j;
    for (i = 0; i < n; i++) {
        im = load_image_color(paths[i], 0, 0);
        sized = letterbox_image(im, netFloat->w, netFloat->h);
        float *X = sized.data;

        network_predict(netFloat, X);
        network_predict(netHalf, X);

        // Compare layer 105 output (float VS half)
        cuda_pull_array(l105_float.output_gpu, l105_float.output, l105_outputs);
        cuda_pull_array(l105_half.output_half_gpu, l105_half.output_half, l105_outputs);
        l105_cmp = compare_layers_output(l105_float.output, l105_half.output_half, l105_outputs);

        // Compare layer 106 output (float VS half)
        cuda_pull_array(l106_float.output_gpu, l106_float.output, l106_outputs);
        cuda_pull_array(l106_half.output_half_gpu, l106_half.output_half, l106_outputs);
        l106_cmp = compare_layers_output(l106_float.output, l106_half.output_half, l106_outputs);

        // Load image valid labels
        replace_image_to_label(paths[i], label_path);
        box_label *truth = read_boxes(label_path, &num_labels);

        // Print results
        img_name = basename(paths[i]);
        int last_class_id = -1;

        fprintf(outFile, "#%d,%s,objects: %d,", i, img_name, num_labels);
        for (j = 0; j < num_labels; j++) {
            int class_id = truth[j].id;
            if (class_id != last_class_id)
                fprintf(outFile, "%s ", class_names[class_id]);
            last_class_id = class_id;
        }
        fprintf(outFile, "\n");
        fprintf(outFile, ",LAYER 105,LAYER 106,\n");
        fprintf(outFile, "AVG,%.2f%%,%.2f%%,%.2f\n", l105_cmp->err_avg*100, l106_cmp->err_avg*100, l106_cmp->err_avg/l105_cmp->err_avg);
        fprintf(outFile, "MAX,%.1f%%,%.1f%%,\n", l105_cmp->err_max*100, l106_cmp->err_max*100);
        fprintf(outFile, "MIN,%f%%,%f%%,\n", l105_cmp->err_min*100, l106_cmp->err_min*100);
        for (j = 0; j < LAYERS_OUT_COMP_INTVL_LIMITS+1; j++) {
            float inf = (j-1 < 0) ? 0.0 : l105_cmp->intvl_limits[j-1];
            float sup = (j >= LAYERS_OUT_COMP_INTVL_LIMITS) ? INFINITY : l105_cmp->intvl_limits[j]; 
            fprintf(outFile, "[%.0f%%..%.0f%%],", inf*100, sup*100);
            fprintf(outFile, "%.2f%%,", (float)l105_cmp->intvl_counts[j]/l105_outputs*100);
            fprintf(outFile, "%.2f%%,\n", (float)l106_cmp->intvl_counts[j]/l106_outputs*100);
        }
        fprintf(outFile, "\n");

        free(l105_cmp);
        free(l106_cmp);
        free(truth);
        free_image(im);
        free_image(sized);

        fprintf(stderr, "\r%d ", i+1);
    }

    fclose(outFile);

    printf("\nTotal time for %d frame(s): %f seconds\n", n, what_time_is_it_now() - t1);
    free_network(netFloat);
    free_network(netHalf);
}

/* Test 7
 * Description: Compare outputs of all layers of YOLOv3. Get relative error between full-float network and mix-precision network.
 * Call: ./darknet detector rtest 7 <mixnet-cfgfile> [-n <images>]
 * Optionals: -n <images> - number of images from COCO dataset (max: 4999, min: 2, default: 4999)
 * Output: CSV file with average error for each layer output of each frame/image
 * IMPORTANT: REAL=float (on Makefile)
 */
void test7(char *cfgfile_mix, int n) {
    if (REAL != FLOAT) {
        printf("Default REAL must be FLOAT (REAL=float on Makefile)!\n");
        return;
    }

    char *cfgfile_float = (char*)"cfg/yolov3.cfg";
    char *weightfile = (char*)"../yolov3.weights";

    network *netFloat = load_network(cfgfile_float, weightfile, 0);
    set_batch_network(netFloat, 1);
    network *netMix = load_network(cfgfile_mix, weightfile, 0);
    set_batch_network(netMix, 1);
    srand(2222222);

    char *valid_images = (char*)"../coco_test/5k.txt";
    list *plist = get_paths(valid_images);
    char **paths = (char**)list_to_array(plist);
    image im, sized;
    if (n > plist->size) {
        fprintf(stderr, "Argument 'n' cannot be greater than %d!", plist->size);
        return;
    }

    FILE *outFile = fopen("layersOut-mixXfloat.csv", "w");
    if (!outFile) {
        perror("Could not create output file: ");
        return;
    }

    int i, j;
    float *relErrArray = (float*)calloc(20000000, sizeof(float));
    float relErrAvgMatrix[netFloat->n][n];

    double t1 = what_time_is_it_now();
    
    for (i = 0; i < n; i++) {
        im = load_image_color(paths[i], 0, 0);
        sized = letterbox_image(im, netFloat->w, netFloat->h);
        float *X = sized.data;

        network_predict(netFloat, X);
        network_predict(netMix, X);

        for (j = 0; j < netFloat->n; j++) {
            layer lF = netFloat->layers[j];
            layer lM = netMix->layers[j];
            int layerOutputs = lF.outputs;

            if (IS_MIX_PRECISION_HALF_LAYER(lM.real_type))
                relative_error_gpu(lF.output_gpu, lM.output_half_gpu, layerOutputs, netFloat->hold_input_gpu);
            else
                relative_error_gpu(lF.output_gpu, lM.output_gpu, layerOutputs, netFloat->hold_input_gpu);

            cuda_pull_array(netFloat->hold_input_gpu, relErrArray, layerOutputs);
            
            relErrAvgMatrix[j][i] = array_average(relErrArray, layerOutputs);
        }

        free_image(im);
        free_image(sized);

        fprintf(stderr, "\r%d ", i+1);
    }

    // Print to CSV file
    fprintf(outFile, ",TYPE");
    for (i = 0; i < n; i++)
        fprintf(outFile, ",frame %3d", i);
    fprintf(outFile, "\n");
    for (i = 0; i < netFloat->n; i++) {
        fprintf(outFile, "layer %3d,%s", i, get_real_string(netMix->layers[i].real_type));
        for (j = 0; j < n; j++)
            fprintf(outFile, ",%.5f", relErrAvgMatrix[i][j]);
        fprintf(outFile, "\n");
    }
    fclose(outFile);

    float l81Sum = 0, l82Sum = 0, l93Sum = 0, l94Sum = 0, l105Sum = 0, l106Sum = 0;
    for (i = 0; i < n; i++) {
        l81Sum += relErrAvgMatrix[81][i];
        l82Sum += relErrAvgMatrix[82][i];
        l93Sum += relErrAvgMatrix[93][i];
        l94Sum += relErrAvgMatrix[94][i];
        l105Sum += relErrAvgMatrix[105][i];
        l106Sum += relErrAvgMatrix[106][i];
    }
    free(relErrArray);

    printf("\n\nAverage errors:\n");
    printf("Layer  81 (CONV): %.2f%%\n", l81Sum/n*100);
    printf("Layer  82 (YOLO): %.2f%%\n", l82Sum/n*100);
    printf("Layer  93 (CONV): %.2f%%\n", l93Sum/n*100);
    printf("Layer  94 (YOLO): %.2f%%\n", l94Sum/n*100);
    printf("Layer 105 (CONV): %.2f%%\n", l105Sum/n*100);
    printf("Layer 106 (YOLO): %.2f%%\n", l106Sum/n*100);

    int halfLayers = 0;
    for (i = 0; i < netMix->n; i++) {
        if (netMix->layers[i].real_type == HALF)
            halfLayers++;
    }

    printf("\nMixed-Precision Network:\n");
    printf("FLOAT Layers: %d\n", netMix->n - halfLayers);
    printf("HALF Layers: %d\n", halfLayers);

    printf("\nTotal time for %d frame(s): %f seconds\n", n, what_time_is_it_now() - t1);
    free_network(netFloat);
    free_network(netMix);
}

/* Test 8
 * Description: Compare outputs of some layers of YOLOv3. Get relative error between full-float network and mix-precision network.
 *      The layers to be compared must be specified by parameter.
 * Call: ./darknet detector rtest 8 <mixnet-cfgfile> [-n <images>] [-gpu <gpu-index>]
 * Optionals: -n <images> - number of images from COCO dataset (max: 4999, min: 2, default: 4999)
 *            -gpu <gpu-index> - index of GPU device
 * IMPORTANT: REAL=float (on Makefile)
 */
void test8(char *cfgfile_mix, int n, int *layers, int nlayers, int gpu) {
    if (REAL != FLOAT) {
        printf("Default REAL must be FLOAT (REAL=float on Makefile)!\n");
        return;
    }

    cuda_set_device(gpu);

    char *cfgfile_float = (char*)"cfg/yolov3.cfg";
    char *weightfile = (char*)"../yolov3.weights";

    network *netFloat = load_network(cfgfile_float, weightfile, 0);
    set_batch_network(netFloat, 1);
    network *netMix = load_network(cfgfile_mix, weightfile, 0);
    set_batch_network(netMix, 1);
    srand(2222222);

    char *valid_images = (char*)"../coco_test/5k.txt";
    list *plist = get_paths(valid_images);
    char **paths = (char**)list_to_array(plist);
    if (n > plist->size) {
        fprintf(stderr, "Argument 'n' cannot be greater than %d!", plist->size);
        return;
    }

    int nthreads = 4;
    if (n < 4) nthreads = n;
    image *val = (image*)calloc(nthreads, sizeof(image));
    image *val_resized = (image*)calloc(nthreads, sizeof(image));
    image *buf = (image*)calloc(nthreads, sizeof(image));
    image *buf_resized = (image*)calloc(nthreads, sizeof(image));
    pthread_t *thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = netFloat->w;
    args.h = netFloat->h;
    args.type = LETTERBOX_DATA;

    int i = 0, j, t;
    float *relErrArray = (float*)calloc(20000000, sizeof(float));
    float relErrAvgMatrix[nlayers][n];

    double t1 = what_time_is_it_now();

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }

    for (i = nthreads; i < n + nthreads; i += nthreads) {
        // > Load images data
        for (t = 0; t < nthreads && i + t - nthreads < n; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < n; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }

        for (t = 0; t < nthreads && i + t - nthreads < n; ++t) {
            int image_index = i + t - nthreads;
            float *X = val_resized[t].data;

            network_predict(netFloat, X);
            network_predict(netMix, X);

            for (j = 0; j < nlayers; j++) {
                int layerIndex = layers[j];
                layer lF = netFloat->layers[layerIndex];
                layer lM = netMix->layers[layerIndex];
                int layerOutputs = lF.outputs;

                if (IS_MIX_PRECISION_HALF_LAYER(lM.real_type))
                    relative_error_gpu(lF.output_gpu, lM.output_half_gpu, layerOutputs, netFloat->hold_input_gpu);
                else
                    relative_error_gpu(lF.output_gpu, lM.output_gpu, layerOutputs, netFloat->hold_input_gpu);

                cuda_pull_array(netFloat->hold_input_gpu, relErrArray, layerOutputs);
                
                relErrAvgMatrix[j][image_index] = array_average(relErrArray, layerOutputs);
            }

            free_image(val[t]);
            free_image(val_resized[t]);

            fprintf(stderr, "\r%d ", image_index+1);
        }
    }

    printf("\n\nAverage errors:\n");
    float layersErrorSum[nlayers] = {};
    for (j = 0; j < nlayers; j++) {
        for (i = 0; i < n; i++) {
            layersErrorSum[j] += relErrAvgMatrix[j][i];
        }
        printf("Layer %3d: %f%%\n", layers[j], layersErrorSum[j]/n*100);
    }
    free(relErrArray);

    int halfLayers = 0;
    for (i = 0; i < netMix->n; i++) {
        if (netMix->layers[i].real_type == HALF)
            halfLayers++;
    }

    printf("\nMixed-Precision Network:\n");
    printf("FLOAT Layers: %d\n", netMix->n - halfLayers);
    printf("HALF Layers: %d\n", halfLayers);

    printf("\nTotal time for %d frame(s): %f seconds\n", n, what_time_is_it_now() - t1);
    free_network(netFloat);
    free_network(netMix);
}

/* Test 9
 * Description: Compare outputs of some layers of YOLOv3. Get relative error between full-float network and mix-precision network.
 *      The layers to be compared must be specified by parameter.
 * Call: ./darknet detector rtest 9 <mixnet-cfgfile> [-n <images>] [-gpu <gpu-index>]
 * Optionals: -n <images> - number of images from COCO dataset (max: 4999, min: 2, default: 4999)
 *            -gpu <gpu-index> - index of GPU device
 * Output: CSV file with results for each frame (average error in each layer, frame truth labels, ...)
 * IMPORTANT: REAL=float (on Makefile)
 */
void test9(char *cfgfile_mix, int n, int *layers, int nlayers, int gpu) {
    if (REAL != FLOAT) {
        printf("Default REAL must be FLOAT (REAL=float on Makefile)!\n");
        return;
    }

    cuda_set_device(gpu);

    char *cfgfile_float = (char*)"cfg/yolov3.cfg";
    char *weightfile = (char*)"../yolov3.weights";

    network *netFloat = load_network(cfgfile_float, weightfile, 0);
    set_batch_network(netFloat, 1);
    network *netMix = load_network(cfgfile_mix, weightfile, 0);
    set_batch_network(netMix, 1);
    srand(2222222);

    char *valid_images = (char*)"../coco_test/5k.txt";
    list *plist = get_paths(valid_images);
    char **paths = (char**)list_to_array(plist);
    if (n > plist->size) {
        fprintf(stderr, "Argument 'n' cannot be greater than %d!", plist->size);
        return;
    }

    char *datacfg = (char*)"cfg/coco.data";
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, (char*)"names", (char *)"data/names.list");
    char **class_names = get_labels(name_list);

    int nthreads = 4;
    if (n < 4) nthreads = n;
    image *val = (image*)calloc(nthreads, sizeof(image));
    image *val_resized = (image*)calloc(nthreads, sizeof(image));
    image *buf = (image*)calloc(nthreads, sizeof(image));
    image *buf_resized = (image*)calloc(nthreads, sizeof(image));
    pthread_t *thr = (pthread_t*)calloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = netFloat->w;
    args.h = netFloat->h;
    args.type = LETTERBOX_DATA;

    int i = 0, j, t, k;
    float *relErrArray = (float*)calloc(20000000, sizeof(float));
    float relErrAvgMatrix[n][nlayers];

    double t1 = what_time_is_it_now();

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }

    for (i = nthreads; i < n + nthreads; i += nthreads) {
        // > Load images data
        for (t = 0; t < nthreads && i + t - nthreads < n; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < n; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }

        for (t = 0; t < nthreads && i + t - nthreads < n; ++t) {
            int image_index = i + t - nthreads;
            float *X = val_resized[t].data;

            network_predict(netFloat, X);
            network_predict(netMix, X);

            for (j = 0; j < nlayers; j++) {
                int layerIndex = layers[j];
                layer lF = netFloat->layers[layerIndex];
                layer lM = netMix->layers[layerIndex];
                int layerOutputs = lF.outputs;

                if (IS_MIX_PRECISION_HALF_LAYER(lM.real_type))
                    relative_error_gpu(lF.output_gpu, lM.output_half_gpu, layerOutputs, netFloat->hold_input_gpu);
                else
                    relative_error_gpu(lF.output_gpu, lM.output_gpu, layerOutputs, netFloat->hold_input_gpu);

                cuda_pull_array(netFloat->hold_input_gpu, relErrArray, layerOutputs);
                
                relErrAvgMatrix[image_index][j] = array_average(relErrArray, layerOutputs);
            }

            free_image(val[t]);
            free_image(val_resized[t]);

            fprintf(stderr, "\r%d ", image_index+1);
        }
    }

    char outFileName[50];
    snprintf(outFileName, sizeof(outFileName), "results-%d.csv", gpu);
    FILE *outFile = fopen(outFileName, "w");
    if (!outFile) {
        perror("Could not create output file: ");
        return;
    }

    // Print results to CSV file
    // Print header: Frame Index | Frame File | Layer X Err | Layer ... Err | Layer Z Err |
    //                 | # Objects | Avg Objs Area | StdDev Objs Area | Min Obj Area | Max Obj Area | Objs Classes
    fprintf(outFile, "FRAME INDEX;FRAME FILE;");
    for (i = 0; i < nlayers; i++)
        fprintf(outFile, "LAYER %d;", layers[i]);
    fprintf(outFile, "NUMBER OF OBJS;AVG OBJS AREA;STDEV OBJS AREA;MIN OBJ AREA;MAX OBJ AREA;OBJECTS\n");

    box_label *truth;
    LabelsMetrics metrics;
    char *imgName, labelPath[1000];
    float *frameLayersErrors;
    int numLabels = 0;

    for (i = 0; i < n; i++) {
        // Load image truth labels
        replace_image_to_label(paths[i], labelPath);
        truth = read_boxes(labelPath, &numLabels);
        metrics = calcLabelsMetrics(truth, numLabels);
        imgName = basename(paths[i]);
        frameLayersErrors = relErrAvgMatrix[i];

        if (numLabels == 0) continue;

        // Print results for frame 'i'
        fprintf(outFile, "%d;", i);                 // Frame index 
        fprintf(outFile, "%s;", imgName);           // Frame file 
        for (j = 0; j < nlayers; j++)               // Layer X...Z Error
            fprintf(outFile, "%f;", frameLayersErrors[j]);
        fprintf(outFile, "%d;", numLabels);         // Number of objects
        fprintf(outFile, "%f;", metrics.avgArea);   // Average objects area
        fprintf(outFile, "%f;", metrics.stdevArea); // Standard deviation of objects area
        fprintf(outFile, "%f;", metrics.minArea);   // Min object area
        fprintf(outFile, "%f;", metrics.maxArea);   // Max object area
        for (k = 0; k < numLabels; k++)             // Objects classes
            fprintf(outFile, "%s,", class_names[truth[k].id]);
        fprintf(outFile, "\n");
        free(truth);
    }

    fclose(outFile);

    printf("\nTotal time for %d frame(s): %f seconds\n", n, what_time_is_it_now() - t1);
    free_network(netFloat);
    free_network(netMix);
    free(relErrArray);
}

/***
 * Test 10
 * Description: Aiming fault injection, this routine predicts all frames listed on file whose path 
 *              is given by 'frameslistfile' and, for each frame, outputs the following data about each
 *              detected box: -> class, x, y, width, height
 * Call: ./darknet detector rtest 10 <cfgfile> <weightsfile> <frameslistfile> [-thresh <threshold>] [-timedebug 0|1]
 * Optionals: -thresh <threshold> - All detections with objectness under <threshold> will be droped (max: 1, min: 0, default: 0.5)
 *            -timedebug 0|1 - Whether the routine will print execution times (default: 0)
 */
void test10(char *cfgfile, char *weightsfile, char *frameslistfile, float thresh, int timedebug) {
    double t0 = what_time_is_it_now();

    // Load frames list
    list *framesList = get_paths(frameslistfile);
    char **framesPaths = (char**)list_to_array(framesList);
    int nframes = framesList->size;

    // Load network
    network *net = load_network(cfgfile, weightsfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    // Aux vars
    layer l = net->layers[net->n - 1];
    float hier_thresh = 0.5, nms = 0.45;
    detection *detections, det;
    int i, j, nboxes, Class;
    image im, sized;
    float *X, BoxX, BoxY, BoxW, BoxH, Objectness, Prob;

    if (timedebug) fprintf(stderr, "Load net time: %f s\n\n", what_time_is_it_now() - t0);

    for (i = 0; i < nframes; i++) {
        // Load and resize frame 'i'
        im = load_image_color(framesPaths[i], 0, 0);
        sized = letterbox_image(im, net->w, net->h);
        X = sized.data;

        // Predict
        network_predict(net, X);

        // Get detections
        nboxes = 0;
        detections = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, 1);
        if (nms) do_nms_sort(detections, nboxes, l.classes, nms);

        // Print detections
        for (j = 0; j < nboxes; j++) {
            det = detections[j];
            Objectness = det.objectness;
            BoxX = det.bbox.x;
            BoxY = det.bbox.y;
            BoxW = det.bbox.w;
            BoxH = det.bbox.h;
            for (Class = 0; Class < l.classes; Class++) {
                Prob = det.prob[Class];
                if (Prob > thresh) {
                    // Class,X,Y,W,H,Objectness,Prob;
                    printf("%d,%f,%f,%f,%f,%f,%f;",Class,BoxX,BoxY,BoxW,BoxH,Objectness,Prob);
                }
            }
        }
        printf("\n");

        // Free data
        free_image(im);
        free_image(sized);
        free_detections(detections, nboxes);
    }

    free_network(net);

    if (timedebug) fprintf(stderr, "Total execution time: %f s\n\n", what_time_is_it_now() - t0);
}

void run_rtest(int testID, int argc, char **argv) {
    if (testID == 1) {
        char *cfgfile = argv[4];
        char *weightfile = argv[5];
        char *filename = argv[6];
        int n = find_int_arg(argc, argv, (char*)"-n", 1);
        int print = find_int_arg(argc, argv, (char*)"-print", 1);
        test1(cfgfile, weightfile, filename, n, print);
    } else if (testID == 2) {
        char *cfgfile = argv[4];
        char *weightfile = argv[5];
        int n = find_int_arg(argc, argv, (char*)"-n", 4999);
        test2(cfgfile, weightfile, n);
    } else if (testID == 3) {
        char *cfgfile = argv[4];
        char *weightfile = argv[5];
        char *filename = argv[6];
        int n = find_int_arg(argc, argv, (char*)"-n", 50);
        test3(cfgfile, weightfile, filename, n);
    } else if (testID == 4) {
        char *cfgfile = argv[4];
        char *weightfile = argv[5];
        char *inputFile = argv[6];
        int layerIndex = atoi(argv[7]);
        char *outputFile = argv[8];
        int bin = find_int_arg(argc, argv, (char*)"-bin", 1);
        test4(cfgfile, weightfile, inputFile, layerIndex, outputFile, bin);
    } else if (testID == 5) {
        char *inputFile1 = argv[4];
        char *inputFile2 = argv[5];
        test5(inputFile1, inputFile2);
    } else if (testID == 6) {
        int n = find_int_arg(argc, argv, (char*)"-n", 4999);
        test6(n);
    } else if (testID == 7) {
        char *cfgfile_mix = argv[4];
        int n = find_int_arg(argc, argv, (char*)"-n", 4999);
        test7(cfgfile_mix, n);
    } else if (testID == 8) {
        char *cfgfile_mix = argv[4];
        int n = find_int_arg(argc, argv, (char*)"-n", 4999);
        int gpu = find_int_arg(argc, argv, (char*)"-gpu", 0);
        int layers[] = { 106 };
        test8(cfgfile_mix, n, layers, sizeof(layers)/sizeof(int), gpu);
    } else if (testID == 9) {
        char *cfgfile_mix = argv[4];
        int n = find_int_arg(argc, argv, (char*)"-n", 4999);
        int gpu = find_int_arg(argc, argv, (char*)"-gpu", 0);
        int layers[] = { 82, 94, 106 };
        test9(cfgfile_mix, n, layers, sizeof(layers)/sizeof(int), gpu);
    }  else if (testID == 10) {
        char *cfgfile = argv[4];
        char *weightsfile = argv[5];
        char *frameslistfile = argv[6];
        float thresh = find_float_arg(argc, argv, (char*)"-thresh", .5);
        int timedebug = find_int_arg(argc, argv, (char*)"-timedebug", 0);
        test10(cfgfile, weightsfile, frameslistfile, thresh, timedebug);
    } else {
        printf("Invalid test ID!\n");
    }
}