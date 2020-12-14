#ifndef TREE_H
#define TREE_H
#include "darknet.h"

int hierarchy_top_prediction(real *predictions, tree *hier, float thresh, int stride);
float get_hierarchy_probability(real *x, tree *hier, int c, int stride);

#endif
