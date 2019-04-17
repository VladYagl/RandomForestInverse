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
    double error;

    comp(char *type, double error = 0.00): error(error) {
        is_min = strcmp(type, "min") == 0;
    }

    double inc(double value) {
        return value + (is_min ? -1 : +1) * abs(value * error);
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
