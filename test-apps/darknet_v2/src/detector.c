#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "blas.h"

#include "args.h"

#include "log_processing.h"

//my_second
#include "helpful.h"

static int coco_ids[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
		18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
		39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
		58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
		80, 81, 82, 84, 85, 86, 87, 88, 89, 90 };

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus,
		int ngpus, int clear) {
	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	char *backup_directory = option_find_str(options, "backup", "/backup/");

	srand(time(0));
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	float avg_loss = -1;
	network *nets = calloc(ngpus, sizeof(network));

	srand(time(0));
	int seed = rand();
	int i;
	for (i = 0; i < ngpus; ++i) {
		srand(seed);
#ifdef GPU
		cuda_set_device(gpus[i]);
#endif
		nets[i] = parse_network_cfg(cfgfile);
		if (weightfile) {
			load_weights(&nets[i], weightfile);
		}
		if (clear)
			*nets[i].seen = 0;
		nets[i].learning_rate *= ngpus;
	}
	srand(time(0));
	network net = nets[0];

	int imgs = net.batch * net.subdivisions * ngpus;
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate,
			net.momentum, net.decay);
	data train, buffer;

	layer l = net.layers[net.n - 1];

	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
	//int N = plist->size;
	char **paths = (char **) list_to_array(plist);

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.jitter = jitter;
	args.num_boxes = l.max_boxes;
	args.d = &buffer;
	args.type = DETECTION_DATA;
	args.threads = 8;

	args.angle = net.angle;
	args.exposure = net.exposure;
	args.saturation = net.saturation;
	args.hue = net.hue;

	pthread_t load_thread = load_data(args);
	clock_t time;
	int count = 0;
	//while(i*imgs < N*120){
	while (get_current_batch(net) < net.max_batches) {
		if (l.random && count++ % 10 == 0) {
			printf("Resizing\n");
			int dim = (rand() % 10 + 10) * 32;
			if (get_current_batch(net) + 200 > net.max_batches)
				dim = 608;
			//int dim = (rand() % 4 + 16) * 32;
			printf("%d\n", dim);
			args.w = dim;
			args.h = dim;

			pthread_join(load_thread, 0);
			train = buffer;
			free_data(train);
			load_thread = load_data(args);

			for (i = 0; i < ngpus; ++i) {
				resize_network(nets + i, dim, dim);
			}
			net = nets[0];
		}
		time = clock();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		/*
		 int k;
		 for(k = 0; k < l.max_boxes; ++k){
		 box b = float_to_box(train.y.vals[10] + 1 + k*5);
		 if(!b.x) break;
		 printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
		 }
		 */
		/*
		 int zz;
		 for(zz = 0; zz < train.X.cols; ++zz){
		 image im = float_to_image(net.w, net.h, 3, train.X.vals[zz]);
		 int k;
		 for(k = 0; k < l.max_boxes; ++k){
		 box b = float_to_box(train.y.vals[zz] + k*5);
		 printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
		 draw_bbox(im, b, 1, 1,0,0);
		 }
		 show_image(im, "truth11");
		 cvWaitKey(0);
		 save_image(im, "truth11");
		 }
		 */

		printf("Loaded: %lf seconds\n", sec(clock() - time));

		time = clock();
		float loss = 0;
#ifdef GPU
		if(ngpus == 1) {
			loss = train_network(net, train);
		} else {
			loss = train_networks(nets, ngpus, train, 4);
		}
#else
		loss = train_network(net, train);
#endif
		if (avg_loss < 0)
			avg_loss = loss;
		avg_loss = avg_loss * .9 + loss * .1;

		i = get_current_batch(net);
		printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n",
				get_current_batch(net), loss, avg_loss, get_current_rate(net),
				sec(clock() - time), i * imgs);
		if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
#ifdef GPU
			if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
#ifdef GPU
	if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}

static int get_coco_image_id(char *filename) {
	char *p = strrchr(filename, '_');
	return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, box *boxes, float **probs,
		int num_boxes, int classes, int w, int h) {
	int i, j;
	int image_id = get_coco_image_id(image_path);
	for (i = 0; i < num_boxes; ++i) {
		float xmin = boxes[i].x - boxes[i].w / 2.;
		float xmax = boxes[i].x + boxes[i].w / 2.;
		float ymin = boxes[i].y - boxes[i].h / 2.;
		float ymax = boxes[i].y + boxes[i].h / 2.;

		if (xmin < 0)
			xmin = 0;
		if (ymin < 0)
			ymin = 0;
		if (xmax > w)
			xmax = w;
		if (ymax > h)
			ymax = h;

		float bx = xmin;
		float by = ymin;
		float bw = xmax - xmin;
		float bh = ymax - ymin;

		for (j = 0; j < classes; ++j) {
			if (probs[i][j])
				fprintf(fp,
						"{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n",
						image_id, coco_ids[j], bx, by, bw, bh, probs[i][j]);
		}
	}
}

void print_detector_detections(FILE **fps, char *id, box *boxes, float **probs,
		int total, int classes, int w, int h) {
	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = boxes[i].x - boxes[i].w / 2. + 1;
		float xmax = boxes[i].x + boxes[i].w / 2. + 1;
		float ymin = boxes[i].y - boxes[i].h / 2. + 1;
		float ymax = boxes[i].y + boxes[i].h / 2. + 1;

		if (xmin < 1)
			xmin = 1;
		if (ymin < 1)
			ymin = 1;
		if (xmax > w)
			xmax = w;
		if (ymax > h)
			ymax = h;

		for (j = 0; j < classes; ++j) {
			if (probs[i][j])
				fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j], xmin,
						ymin, xmax, ymax);
		}
	}
}

void print_imagenet_detections(FILE *fp, int id, box *boxes, float **probs,
		int total, int classes, int w, int h) {
	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = boxes[i].x - boxes[i].w / 2.;
		float xmax = boxes[i].x + boxes[i].w / 2.;
		float ymin = boxes[i].y - boxes[i].h / 2.;
		float ymax = boxes[i].y + boxes[i].h / 2.;

		if (xmin < 0)
			xmin = 0;
		if (ymin < 0)
			ymin = 0;
		if (xmax > w)
			xmax = w;
		if (ymax > h)
			ymax = h;

		for (j = 0; j < classes; ++j) {
			int class = j;
			if (probs[i][class])
				fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1,
						probs[i][class], xmin, ymin, xmax, ymax);
		}
	}
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile,
		char *outfile) {
	int j;
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.list");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char *prefix = option_find_str(options, "results", "results");
	char **names = get_labels(name_list);
	char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf)
		map = read_map(mapf);

	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 2);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	list *plist = get_paths(valid_images);
	char **paths = (char **) list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;

	char buff[1024];
	char *type = option_find_str(options, "eval", "voc");
	FILE *fp = 0;
	FILE **fps = 0;
	int coco = 0;
	int imagenet = 0;
	if (0 == strcmp(type, "coco")) {
		if (!outfile)
			outfile = "coco_results";
		snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
		fp = fopen(buff, "w");
		fprintf(fp, "[\n");
		coco = 1;
	} else if (0 == strcmp(type, "imagenet")) {
		if (!outfile)
			outfile = "imagenet-detection";
		snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
		fp = fopen(buff, "w");
		imagenet = 1;
		classes = 200;
	} else {
		if (!outfile)
			outfile = "comp4_det_test_";
		fps = calloc(classes, sizeof(FILE *));
		for (j = 0; j < classes; ++j) {
			snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
			fps[j] = fopen(buff, "w");
		}
	}

	box *boxes = calloc(l.w * l.h * l.n, sizeof(box));
	float **probs = calloc(l.w * l.h * l.n, sizeof(float *));
	for (j = 0; j < l.w * l.h * l.n; ++j)
		probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i = 0;
	int t;

	float thresh = .005;
	float nms = .45;

	int nthreads = 4;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	image input = make_image(net.w, net.h, net.c * 2);

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	//args.type = IMAGE_DATA;
	args.type = LETTERBOX_DATA;

	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	time_t start = time(0);
	for (i = nthreads; i < m + nthreads; i += nthreads) {
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads && i + t < m; ++t) {
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			char *path = paths[i + t - nthreads];
			char *id = basecfg(path);
			copy_cpu(net.w * net.h * net.c, val_resized[t].data, 1, input.data,
					1);
			flip_image(val_resized[t]);
			copy_cpu(net.w * net.h * net.c, val_resized[t].data, 1,
					input.data + net.w * net.h * net.c, 1);

			network_predict(net, input.data);
			int w = val[t].w;
			int h = val[t].h;
			get_region_boxes(l, w, h, net.w, net.h, thresh, probs, boxes, 0,
					map, .5, 0);
			if (nms)
				do_nms_sort(boxes, probs, l.w * l.h * l.n, classes, nms);
			if (coco) {
				print_cocos(fp, path, boxes, probs, l.w * l.h * l.n, classes, w,
						h);
			} else if (imagenet) {
				print_imagenet_detections(fp, i + t - nthreads + 1, boxes,
						probs, l.w * l.h * l.n, classes, w, h);
			} else {
				print_detector_detections(fps, id, boxes, probs,
						l.w * l.h * l.n, classes, w, h);
			}
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	for (j = 0; j < classes; ++j) {
		if (fps)
			fclose(fps[j]);
	}
	if (coco) {
		fseek(fp, -2, SEEK_CUR);
		fprintf(fp, "\n]\n");
		fclose(fp);
	}
	fprintf(stderr, "Total Detection Time: %f Seconds\n",
			(double) (time(0) - start));
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile,
		char *outfile) {
	int j;
	list *options = read_data_cfg(datacfg);
	char *valid_images = option_find_str(options, "valid", "data/train.list");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char *prefix = option_find_str(options, "results", "results");
	char **names = get_labels(name_list);
	char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf)
		map = read_map(mapf);

	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	list *plist = get_paths(valid_images);
	char **paths = (char **) list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;

	char buff[1024];
	char *type = option_find_str(options, "eval", "voc");
	FILE *fp = 0;
	FILE **fps = 0;
	int coco = 0;
	int imagenet = 0;
	if (0 == strcmp(type, "coco")) {
		if (!outfile)
			outfile = "coco_results";
		snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
		fp = fopen(buff, "w");
		fprintf(fp, "[\n");
		coco = 1;
	} else if (0 == strcmp(type, "imagenet")) {
		if (!outfile)
			outfile = "imagenet-detection";
		snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
		fp = fopen(buff, "w");
		imagenet = 1;
		classes = 200;
	} else {
		if (!outfile)
			outfile = "comp4_det_test_";
		fps = calloc(classes, sizeof(FILE *));
		for (j = 0; j < classes; ++j) {
			snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
			fps[j] = fopen(buff, "w");
		}
	}

	box *boxes = calloc(l.w * l.h * l.n, sizeof(box));
	float **probs = calloc(l.w * l.h * l.n, sizeof(float *));
	for (j = 0; j < l.w * l.h * l.n; ++j)
		probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i = 0;
	int t;

	float thresh = .005;
	float nms = .45;

	int nthreads = 4;
	image *val = calloc(nthreads, sizeof(image));
	image *val_resized = calloc(nthreads, sizeof(image));
	image *buf = calloc(nthreads, sizeof(image));
	image *buf_resized = calloc(nthreads, sizeof(image));
	pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	//args.type = IMAGE_DATA;
	args.type = LETTERBOX_DATA;

	for (t = 0; t < nthreads; ++t) {
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr[t] = load_data_in_thread(args);
	}
	time_t start = time(0);
	for (i = nthreads; i < m + nthreads; i += nthreads) {
		fprintf(stderr, "%d\n", i);
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			pthread_join(thr[t], 0);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads && i + t < m; ++t) {
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = load_data_in_thread(args);
		}
		for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
			char *path = paths[i + t - nthreads];
			char *id = basecfg(path);
			float *X = val_resized[t].data;
			network_predict(net, X);
			int w = val[t].w;
			int h = val[t].h;
			get_region_boxes(l, w, h, net.w, net.h, thresh, probs, boxes, 0,
					map, .5, 0);
			if (nms)
				do_nms_sort(boxes, probs, l.w * l.h * l.n, classes, nms);
			if (coco) {
				print_cocos(fp, path, boxes, probs, l.w * l.h * l.n, classes, w,
						h);
			} else if (imagenet) {
				print_imagenet_detections(fp, i + t - nthreads + 1, boxes,
						probs, l.w * l.h * l.n, classes, w, h);
			} else {
				print_detector_detections(fps, id, boxes, probs,
						l.w * l.h * l.n, classes, w, h);
			}
			free(id);
			free_image(val[t]);
			free_image(val_resized[t]);
		}
	}
	for (j = 0; j < classes; ++j) {
		if (fps)
			fclose(fps[j]);
	}
	if (coco) {
		fseek(fp, -2, SEEK_CUR);
		fprintf(fp, "\n]\n");
		fclose(fp);
	}
	fprintf(stderr, "Total Detection Time: %f Seconds\n",
			(double) (time(0) - start));
}

void validate_detector_recall(char *cfgfile, char *weightfile) {
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n",
			net.learning_rate, net.momentum, net.decay);
	srand(time(0));

	list *plist = get_paths("data/voc.2007.test");
	char **paths = (char **) list_to_array(plist);

	layer l = net.layers[net.n - 1];
	int classes = l.classes;

	int j, k;
	box *boxes = calloc(l.w * l.h * l.n, sizeof(box));
	float **probs = calloc(l.w * l.h * l.n, sizeof(float *));
	for (j = 0; j < l.w * l.h * l.n; ++j)
		probs[j] = calloc(classes, sizeof(float *));

	int m = plist->size;
	int i = 0;

	float thresh = .001;
	float iou_thresh = .5;
	float nms = .4;

	int total = 0;
	int correct = 0;
	int proposals = 0;
	float avg_iou = 0;

	for (i = 0; i < m; ++i) {
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net.w, net.h);
		char *id = basecfg(path);
		network_predict(net, sized.data);
		get_region_boxes(l, sized.w, sized.h, net.w, net.h, thresh, probs,
				boxes, 1, 0, .5, 1);
		if (nms)
			do_nms(boxes, probs, l.w * l.h * l.n, 1, nms);

		char labelpath[4096];
		find_replace(path, "images", "labels", labelpath);
		find_replace(labelpath, "JPEGImages", "labels", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		for (k = 0; k < l.w * l.h * l.n; ++k) {
			if (probs[k][0] > thresh) {
				++proposals;
			}
		}
		for (j = 0; j < num_labels; ++j) {
			++total;
			box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
			float best_iou = 0;
			for (k = 0; k < l.w * l.h * l.n; ++k) {
				float iou = box_iou(boxes[k], t);
				if (probs[k][0] > thresh && iou > best_iou) {
					best_iou = iou;
				}
			}
			avg_iou += best_iou;
			if (best_iou > iou_thresh) {
				++correct;
			}
		}

		fprintf(stderr,
				"%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i,
				correct, total, (float) proposals / (i + 1),
				avg_iou * 100 / total, 100. * correct / total);
		free(id);
		free_image(orig);
		free_image(sized);
	}
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile,
		char *filename, float thresh, float hier_thresh, char *outfile,
		int fullscreen) {
	list *options = read_data_cfg(datacfg);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);

	image **alphabet = load_alphabet();
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(2222222);
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = .4;
	while (1) {
		if (filename) {
			strncpy(input, filename, 256);
		} else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input)
				return;
			strtok(input, "\n");
		}
		image im = load_image_color(input, 0, 0);
		image sized = letterbox_image(im, net.w, net.h);
		//image sized = resize_image(im, net.w, net.h);
		//image sized2 = resize_max(im, net.w);
		//image sized = crop_image(sized2, -((net.w - sized2.w)/2), -((net.h - sized2.h)/2), net.w, net.h);
		//resize_network(&net, sized.w, sized.h);
		layer l = net.layers[net.n - 1];

		box *boxes = calloc(l.w * l.h * l.n, sizeof(box));
		float **probs = calloc(l.w * l.h * l.n, sizeof(float *));
		for (j = 0; j < l.w * l.h * l.n; ++j)
			probs[j] = calloc(l.classes + 1, sizeof(float *));

		printf("valor w * h * n %d valor side * side * n %d\n", l.w * l.h * l.n,
				l.side * l.side * l.n);

		float *X = sized.data;
		time = clock();
		network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input, sec(clock() - time));
		get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, 0,
				0, hier_thresh, 1);
		if (nms)
			do_nms_obj(boxes, probs, l.w * l.h * l.n, l.classes, nms);
		//else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
		draw_detections(im, l.w * l.h * l.n, thresh, boxes, probs, names,
				alphabet, l.classes);
		if (outfile) {
			save_image(im, outfile);
		} else {
			save_image(im, "predictions");
#ifdef OPENCV
			cvNamedWindow("predictions", CV_WINDOW_NORMAL);
			if(fullscreen) {
				cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			}
			show_image(im, "predictions");
			cvWaitKey(0);
			cvDestroyAllWindows();
#endif
		}

		free_image(im);
		free_image(sized);
		free(boxes);
		free_ptrs((void **) probs, l.w * l.h * l.n);
		if (filename)
			break;
	}
}

/**
 * support functions
 * -------------------------------------------------------------------------------------
 */
image *load_all_images_sized(image *img_array, int net_w, int net_h,
		int list_size) {
//      image sized = letterbox_image(im, net.w, net.h);
	int i;
	image *ret = (image*) malloc(sizeof(image) * list_size);
	for (i = 0; i < list_size; i++) {
		ret[i] = letterbox_image(img_array[i], net_w, net_h);
		if (i > 500) {
			update_timestamp_app();
		}

	}
	return ret;
}

image *load_all_images(detection det) {
//  image im = load_image_color(input, 0, 0);
	int i;
	image *ret = (image*) malloc(sizeof(image) * det.plist_size);
	for (i = 0; i < det.plist_size; i++) {
		ret[i] = load_image_color(det.img_names[i], 0, 0);
		if (i > 500) {
			update_timestamp_app();
		}
	}
	return ret;
}

void free_all_images(image *array, int list_size) {
	//          free_image(im);
	int i;
	for (i = 0; i < list_size; i++) {
		free_image(array[i]);
	}
}
//-------------------------------------------------------------------------------------
int set_abft_approach(Args* arg) {
	//#ifdef GPU
	int mr_size = 1;
	switch (arg->abft) {
	case GEMM:
		printf("%s ABFT not implemented yet\n", ABFT_TYPES[arg->abft]);
		exit(-1);
		break;
	case SMART_POOLING:
		set_abft_smartpool(arg->abft);
		break;
	case L1_HARDENING:
		printf("%s ABFT not implemented yet\n", ABFT_TYPES[arg->abft]);
		exit(-1);
		break;
	case L2_HARDENING:
		printf("%s ABFT not implemented yet\n", ABFT_TYPES[arg->abft]);
		exit(-1);
		break;
	case TRAINED_WEIGHTS:
		printf("%s ABFT not implemented yet\n", ABFT_TYPES[arg->abft]);
		exit(-1);
		break;
	case SMART_DMR:
		mr_size = 2;
		break;
	case SMART_TMR:
		mr_size = 3;
		break;
	default:
		printf("No ABFT was set\n");
		break;
	}
	return mr_size;
}

/**
 * Function created only for radiation test only
 * args is an Args
 */
void test_detector_radiation(Args *args) {
	//load all information from the goldfile
	detection gold = load_gold(args);
	gold.network_name = "darknet_v2";
	printf("\nArgs inside detector_radiation\n");
	print_args(*args);

#ifdef GEN_IMG
	//load cfg file
	list *options = read_data_cfg(args->cfg_data);

	// here it takes data/coco.names
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	image **alphabet = load_alphabet();
#endif
	network net = parse_network_cfg(args->config_file);

	if (args->weights) {
		load_weights(&net, args->weights);
	}
	set_batch_network(&net, 1);
	srand(2222222);

	int j, i, it;
	float nms = .4;
	//load all images
	const image *im_array = load_all_images(gold);

	const image *im_array_sized = load_all_images_sized(im_array, net.w, net.h,
			gold.plist_size);

	//if abft is set these parameters will also be set
	error_return max_pool_errors;
	init_error_return(&max_pool_errors);
	int mr_size;
	//  set abft
	if (args->abft >= 0 && args->abft < MAX_ABFT_TYPES) {
#ifdef GPU
		mr_size = set_abft_approach(args);
#endif
	}

//  alloc once and clear at each iteration
	layer l = net.layers[net.n - 1];
	box *boxes = calloc(l.w * l.h * l.n, sizeof(box));
	float **probs = calloc(l.w * l.h * l.n, sizeof(float *));
	for (j = 0; j < l.w * l.h * l.n; ++j)
		probs[j] = calloc(l.classes + 1, sizeof(float *));

	//need to allocate layers arrays
	alloc_gold_layers_arrays(&gold, &net);
	// this loop will iterate all iteration on args * image_size
	int overall_errors = 0;
	for (i = 0; i < args->iterations; i++) {
		for (it = 0; it < gold.plist_size; it++) {
			image im = im_array[it];
			image sized = im_array_sized[it];

			float *X = sized.data;

			double time = mysecond();
			//This is the detection
			start_iteration_app();

			network_predict(net, X);

			get_region_boxes(l, im.w, im.h, net.w, net.h, args->thresh, probs,
					boxes, 0, 0, args->hier_thresh, 1);

			if (nms)
				do_nms_obj(boxes, probs, l.w * l.h * l.n, l.classes, nms);

			end_iteration_app();
			time = mysecond() - time;
//      here we test if any error happened
//          if shit happened we log
			double time_cmp = mysecond();
//          void compare(prob_array gold, float **f_probs, box *f_boxes, int num,
//                  int classes, int img, int save_layer, network net, int test_iteration)

#ifdef GPU
			//before compare copy maxpool err detection values
			//smart pooling
			if (args->abft == 2) {
				get_and_reset_error_detected_values(&max_pool_errors);
			}
#endif
			int error_count = compare(&gold, probs, boxes, l.w * l.h * l.n, l.classes, it,
					args->save_layers, i, args->img_list_path, max_pool_errors, im, 1);
			time_cmp = mysecond() - time_cmp;

			printf(
					"Iteration %d - image %d predicted in %f seconds. Comparisson in %f seconds.\n",
					i, it, time, time_cmp);

			if ((overall_errors +=  error_count) > MAX_ERROR_COUNT){
				const char error_string[1000];
				sprintf(error_string, "%d ERRORS DETECTED, WHITCH IS MUCH MORE THAN %d. FINISHING APPLICATION\n", overall_errors, MAX_ERROR_COUNT);
				error(error_string);
			}

//########################################

#ifdef GEN_IMG
			draw_detections(im, l.w * l.h * l.n, args->thresh, boxes, probs, names,
					alphabet, l.classes);
			char temp[10];
			sprintf(temp, "pred%d", it);
			save_image(im, temp);
#endif

			clear_boxes_and_probs(boxes, probs, l.w * l.h * l.n, l.classes);
		}
	}

	//free the memory
	free_ptrs((void **) probs, l.w * l.h * l.n);
	free(boxes);
	delete_detection_var(&gold, args);

	free_all_images(im_array, gold.plist_size);
	free_all_images(im_array_sized, gold.plist_size);

	//free smartpool errors
	free_error_return(&max_pool_errors);
#ifdef GPU
	free_err_detected();
#endif
}

void test_detector_generate(Args *args) {
	// first I nee to treat all image files
	int img_list_size = 0;
	char **img_list = get_image_filenames(args->img_list_path, &img_list_size);

#ifdef GEN_IMG
	//load cfg file
	list *options = read_data_cfg(args->cfg_data);

	// here it takes data/coco.names
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	image **alphabet = load_alphabet();
#endif

	network net = parse_network_cfg(args->config_file);
	if (args->weights) {
		load_weights(&net, args->weights);
	}
	set_batch_network(&net, 1);
	srand(2222222);

	//output gold
	layer l = net.layers[net.n - 1];
	int total = l.w * l.h * l.n;
	int classes = l.classes;
	FILE *output_file = fopen(args->gold_inout, "w+");
	if (output_file) {
//      writing all parameters for test execution
//      thresh hier_tresh img_list_size img_list_path config_file config_data model weights total classes

		fprintf(output_file, "%f;%f;%d;%s;%s;%s;%s;%s;%d;%d;\n", args->thresh,
				args->hier_thresh, img_list_size, args->img_list_path,
				args->config_file, args->cfg_data, args->model, args->weights,
				total, classes);
	} else {
		fprintf(stderr, "GOLD OPENING ERROR");
		exit(-1);
	}
	printf("total %d and other %d\n", l.side * l.side * l.n, l.h * l.n * l.w);

	//---------------------------------------

	int j;
	float nms = .4;

	detection gold_to_save;
	gold_to_save.network_name = "darknet_v2";
	if (args->save_layers)
		alloc_gold_layers_arrays(&gold_to_save, &net);

	// this loop will iterate for all images
	int it;
	for (it = 0; it < img_list_size; it++) {
		printf("generating gold for: %s\n", img_list[it]);
		image im = load_image_color(img_list[it], 0, 0);
		image sized = letterbox_image(im, net.w, net.h);
		l = net.layers[net.n - 1];

		box *boxes = calloc(l.w * l.h * l.n, sizeof(box));
		float **probs = calloc(l.w * l.h * l.n, sizeof(float *));
		for (j = 0; j < l.w * l.h * l.n; ++j)
			probs[j] = calloc(l.classes + 1, sizeof(float *));

		float *X = sized.data;
		network_predict(net, X);
		get_region_boxes(l, im.w, im.h, net.w, net.h, args->thresh, probs,
				boxes, 0, 0, args->hier_thresh, 1);
		if (nms)
			do_nms_obj(boxes, probs, l.w * l.h * l.n, l.classes, nms);

//      must do the same thing that draw_detections
//      but the output will be a gold file (old draw_detections)
//      first write a filename
		fprintf(output_file, "%s;%d;%d;%d;\n", img_list[it], im.h, im.w, im.c);
//      after writes all detection information
//      each box is described as class number, left, top, right, bottom, prob (confidence)
//      save_gold(FILE *fp, char *img, int num, int classes, float **probs,
//              box *boxes)
		save_gold(output_file, img_list[it], l.w * l.h * l.n, l.classes, probs,
				boxes);

		if (args->save_layers)
			save_layer(&gold_to_save, it, 0, "gold", 1, args->img_list_path);

#ifdef GEN_IMG
		draw_detections(im, l.w * l.h * l.n, args->thresh, boxes, probs, names,
				alphabet, l.classes);
		char temp[10];
		sprintf(temp, "pred%d", it);
		save_image(im, temp);
#endif

		free_image(im);
		free_image(sized);
		free(boxes);
		free_ptrs((void **) probs, l.w * l.h * l.n);

	}

	//free char** memory
	for (it = 0; it < img_list_size; it++) {
		free(img_list[it]);
	}
	free(img_list);

	//close gold file
	fclose(output_file);
}

void run_detector(int argc, char **argv) {
	char *prefix = find_char_arg(argc, argv, "-prefix", 0);
	float thresh = find_float_arg(argc, argv, "-thresh", .24);
	float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
	int frame_skip = find_int_arg(argc, argv, "-s", 0);
	if (argc < 4) {
		fprintf(stderr,
				"usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n",
				argv[0], argv[1]);
		return;
	}
	char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
	char *outfile = find_char_arg(argc, argv, "-out", 0);
	int *gpus = 0;
	int gpu = 0;
	int ngpus = 0;
	if (gpu_list) {
		printf("%s\n", gpu_list);
		int len = strlen(gpu_list);
		ngpus = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (gpu_list[i] == ',')
				++ngpus;
		}
		gpus = calloc(ngpus, sizeof(int));
		for (i = 0; i < ngpus; ++i) {
			gpus[i] = atoi(gpu_list);
			gpu_list = strchr(gpu_list, ',') + 1;
		}
	} else {
		gpu = gpu_index;
		gpus = &gpu;
		ngpus = 1;
	}

	int clear = find_arg(argc, argv, "-clear");
	int fullscreen = find_arg(argc, argv, "-fullscreen");
	int width = find_int_arg(argc, argv, "-w", 0);
	int height = find_int_arg(argc, argv, "-h", 0);
	int fps = find_int_arg(argc, argv, "-fps", 0);

	char *datacfg = argv[3];
	char *cfg = argv[4];
	char *weights = (argc > 5) ? argv[5] : 0;
	char *filename = (argc > 6) ? argv[6] : 0;
	if (0 == strcmp(argv[2], "test"))
		test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh,
				outfile, fullscreen);
	else if (0 == strcmp(argv[2], "train"))
		train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
	else if (0 == strcmp(argv[2], "valid"))
		validate_detector(datacfg, cfg, weights, outfile);
	else if (0 == strcmp(argv[2], "valid2"))
		validate_detector_flip(datacfg, cfg, weights, outfile);
	else if (0 == strcmp(argv[2], "recall"))
		validate_detector_recall(cfg, weights);
	else if (0 == strcmp(argv[2], "demo")) {
		list *options = read_data_cfg(datacfg);
		int classes = option_find_int(options, "classes", 20);
		char *name_list = option_find_str(options, "names", "data/names.list");
		char **names = get_labels(name_list);
		demo(cfg, weights, thresh, cam_index, filename, names, classes,
				frame_skip, prefix, hier_thresh, width, height, fps,
				fullscreen);
	}
}

