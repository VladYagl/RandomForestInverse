#include <algorithm>
#include <cstdio>
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

    void print(FILE* stream) {
        fprintf(stream, "\n");
        for (size_t i = 0; i < n_features; i++) {
            fprintf(stream, "    %lf\t%lf\n", lower[i], upper[i]);
        }
        fprintf(stream, "\n");
    }

    void copy(const rect& rhs) {
        for (size_t i = 0; i < n_features; i++) {
            lower[i] = rhs.lower[i];
            upper[i] = rhs.upper[i];
        }
    };
};
