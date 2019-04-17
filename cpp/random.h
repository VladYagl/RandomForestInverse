#pragma once

#include <random>

#include "forest.h"

class rnd : public forest {
    vector<std::uniform_real_distribution<double>> dstr;
    std::default_random_engine engine;

public:
    rnd(size_t forest_size, size_t n_features, const comp& cmp)
        : forest(forest_size, n_features, cmp) {
    }

    pair<double, rect> inverse() {
        cerr << "RANDOM" << endl;
        double sum = 0;
        for (auto root : trees) {
            sum += root->shit;
        }
        auto ans = rect(n_features);
        double ans_value = sum;

        for (size_t i = 0; i < n_features; i++) {
            dstr.push_back(std::uniform_real_distribution(limits.lower[i], limits.upper[i]));
        }
        vector<double> value(n_features);

        for (size_t shit = 0; shit < iterations; shit++) {
            for (size_t i = 0; i < n_features; i++) {
                value[i] = dstr[i](engine);
            }
            double result = predict(value);
            if (cmp(result, ans_value)) {
                ans_value = result;
                for (size_t i = 0; i < n_features; i++) {
                    ans.set(i, make_pair(value[i], value[i]));
                }
            }
        }

        return make_pair(ans_value, ans); // I divide by forest size in daddy.cpp
    }
};
