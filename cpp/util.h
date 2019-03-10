#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace std;

struct rect {
    double *lower, *upper;
    size_t n_features;

    rect() {
    }

    rect(int n_features) : n_features(n_features) {
        lower = new double[n_features];
        upper = new double[n_features];
        fill(lower, &lower[n_features], -numeric_limits<double>::infinity());
        fill(upper, &upper[n_features], numeric_limits<double>::infinity());
    }

    void set(int feature, pair<double, double> border) {
        lower[feature] = border.first;
        upper[feature] = border.second;
    }

    void print(FILE *stream) const {
        fprintf(stream, "\n");
        for (size_t i = 0; i < n_features; i++) {
            fprintf(stream, "    %lf\t%lf\n", lower[i], upper[i]);
        }
        fprintf(stream, "\n");
    }

    void copy(const rect &rhs) {
        for (size_t i = 0; i < n_features; i++) {
            lower[i] = rhs.lower[i];
            upper[i] = rhs.upper[i];
        }
    };
};

struct comp {
    bool is_min = false;

    comp(char *type) {
        is_min = strcmp(type, "min") == 0;
    }

    bool operator()(double a, double b) const {
        if (is_min)
            return a < b;
        else
            return a > b;
    }

    double best(double a, double b) const {
        if ((*this)(a, b))
            return a;
        else
            return b;
    }

    double worst(double a, double b) const {
        if ((*this)(a, b))
            return b;
        else
            return a;
    }
};

struct node {
    size_t index, size, depth;
    int feature;
    node *left, *right;
    double threshold, value, shit;
    int left_index, right_index;

    node(size_t index, int feature, double threshold, double value, int left_index, int right_index)
        : index(index),
          feature(feature),
          threshold(threshold),
          value(value),
          shit(value),
          left_index(left_index),
          right_index(right_index) {
        left = nullptr;
        right = nullptr;
        size = 1;
        depth = 1;
    }

    void precalc(const comp &cmp) {
        if (is_leaf()) return;
        left->precalc(cmp);
        right->precalc(cmp);
        value = cmp.best(left->value, right->value);
        shit = cmp.worst(left->shit, right->shit);
        size = left->size + right->size;
        depth = max(left->depth, right->depth) + 1;
    }

    bool is_leaf() const {
        return left == nullptr;
    }
};
