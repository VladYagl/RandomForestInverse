#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include <iostream>

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

    rect(const rect &rhs): n_features(rhs.n_features) {
        lower = new double[n_features];
        upper = new double[n_features];
        copy(rhs);
    };

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

bool intersects(const rect &ths, const rect &rhs) {
    for (size_t i = 0; i < ths.n_features; i++) {
        if (ths.lower[i] > rhs.upper[i] - 1e-8) return false; // >=
        if (ths.upper[i] < rhs.lower[i] + 1e-8) return false; // <=
    }
    return true;
}

typedef pair<double, const rect*> leaf;
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

    bool operator()(const leaf a, const leaf b) const {
        if (is_min)
            return a.first < b.first;
        else
            return a.first > b.first;
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
    size_t feature;
    node *left, *right;
    double threshold, value, shit;
    size_t left_index, right_index;
    size_t n_features;
    vector<leaf> leaves;
    rect area;

    node(size_t index, size_t feature, double threshold, double value, size_t left_index, size_t right_index, size_t n_features)
        : index(index),
          feature(feature),
          threshold(threshold),
          value(value),
          shit(value),
          left_index(left_index),
          right_index(right_index),
          n_features(n_features),
          area(n_features) {
        left = nullptr;
        right = nullptr;
        size = 1;
        depth = 1;
    }

    void precalc(const comp &cmp) {
        if (is_leaf()) {
            leaves.emplace_back(value, &area);
            return;
        }

        left->area.copy(area);
        left->area.upper[feature] = min(left->area.upper[feature], threshold);
        right->area.copy(area);
        right->area.lower[feature] = max(right->area.lower[feature], threshold);
        left->precalc(cmp);
        right->precalc(cmp);

        leaves.reserve(left->leaves.size() + right->leaves.size());
        merge(left->leaves.begin(), left->leaves.end(), right->leaves.begin(), right->leaves.end(), back_inserter(leaves), cmp);

        value = cmp.best(left->value, right->value);
        shit = cmp.worst(left->shit, right->shit);
        size = left->size + right->size;
        depth = max(left->depth, right->depth) + 1;
    }

    bool is_leaf() const {
        return left == nullptr;
    }
};
