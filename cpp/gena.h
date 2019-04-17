#pragma once

#include <random>
#include <set>
#include <unistd.h>

#include "forest.h"

class threshold {
public:
    double split;
    set<size_t> trees;

    threshold(double split, size_t tree) : split(split) {
        trees.insert(tree);
    }

    bool operator<(const threshold& other) const {
        return split < other.split;
    }
};

class gena : public forest {
    vector<vector<threshold>> space;
    vector<size_t> pos;
    vector<double> location;
    std::uniform_real_distribution<double> dice = uniform_real_distribution(0.0, 1.0);
    std::default_random_engine engine;

    double rnd() {
        return dice(engine);
    }

    void divide_tree(node* v, size_t tree) {
        if (!v->is_leaf()) {
            space[v->feature].emplace_back(v->threshold, tree);
            divide_tree(v->left, tree);
            divide_tree(v->right, tree);
        }
    }

    void divide_space() {
        for (size_t i = 0; i < trees.size(); i++) {
            divide_tree(trees[i], i);
        }
        for (auto& a : space) {
            sort(a.begin(), a.end());
            for (size_t i = 0; i < a.size() - 1; i++) {  // TODO: !!! O(n^2) !!!
                while (i < a.size() - 1 && a[i + 1].split - a[i].split < 1e-4 * abs(a[i].split) + 1e-8) { // TODO : epsilon
                    a[i].trees.insert(*a[i + 1].trees.begin());
                    a.erase(a.begin() + i + 1);
                }
            }
        }

        /* for (auto a : space) { */
        /*     for (auto i : a) { */
        /*         cerr << "| | " << i.split; */
        /*         for (auto j : i.trees) { */
        /*             cerr << ' ' << j; */
        /*         } */
        /*         cerr << endl; */
        /*     } */
        /*     cerr << "--------------------------" << endl; */
        /* } */
    }

    void set(size_t feature, size_t pos) {
        if (pos == 0) {
            location[feature] = space[feature].front().split - 1;
        } else if (pos == space[feature].size()) {
            location[feature] = space[feature].back().split + 1;
        } else {
            location[feature] = (space[feature][pos - 1].split + space[feature][pos].split) / 2;
        }
    }

public:
    gena(size_t forest_size, size_t n_features, const comp& cmp)
        : forest(forest_size, n_features, cmp), space(n_features), pos(n_features), location(n_features) {
    }

    pair<double, rect> inverse() {
        cerr.precision(15);
        double best_value = 0;
        for (auto root : trees) {
            best_value += root->shit;
        }
        auto best = location;

        divide_space();

        for (size_t kappa = 0; kappa < 100000; kappa++) {
            for (size_t i = 0; i < n_features; i++) {
                pos[i] = rand() % (space[i].size() + 1);
                set(i, pos[i]);
            }
            vector<double> current;
            for (auto i : trees) {
                current.push_back(i->predict(location)->value);
            }
            double p = 0.75;
            double q = (1 - p) / iterations;
            double ans_value = predict(location);
            cmp.error = 1e-8;

            for (size_t shit = 0; shit < iterations; shit++) {
                size_t feature = rand() % n_features;
                int move;
                if (pos[feature] == space[feature].size()) {
                    move = -1;
                } else if (pos[feature] == 0) {
                    move = +1;
                } else {
                    if (rnd() > 0.5) {
                        move = +1;
                    } else {
                        move = -1;
                    }
                }

                /* cerr << "-------------------------------" << endl; */
                threshold* threshold;
                if (move == +1) {
                    threshold = &space[feature][pos[feature]];
                } else {
                    threshold = &space[feature][pos[feature] - 1];
                }
                /* cerr << "__value__" << endl; */
                /* cerr << location[feature] << endl; */
                /* cerr << threshold->split; */
                /* for (auto i : threshold->trees) { */
                    /* cerr << ' ' << i; */
                /* } */
                /* cerr << endl; */
                set(feature, pos[feature] + move);
                /* cerr << location[feature] << endl; */
                /* cerr << "_________" << endl; */
                double old_value = 0;
                double new_value = 0;
                for (auto tree : threshold->trees) {
                    old_value += current[tree];
                    new_value += trees[tree]->predict(location)->value;
                }
                /* cerr << "old/new: " << old_value << ' ' << new_value << endl; */
                /* for (size_t i = 0; i < forest_size; i++) { */
                /*     cerr << "[" << current[i] << ' ' << trees[i]->predict(location)->value << "]" << endl; */
                /* } */
                if (cmp(cmp.inc(new_value), old_value) == (rnd() < p)) {
                    for (auto tree : threshold->trees) {
                        current[tree] = trees[tree]->predict(location)->value;
                    }
                    pos[feature] += move;
                    ans_value = ans_value - old_value + new_value;
                    /* cerr << "ANS: " << ans_value << ' ' << predict(location) << endl; */
                    cerr << ans_value << endl;
                    /* for (auto i : location) { */
                    /*     cerr << i << '\t'; */
                    /* } */
                    /* cerr << endl; */
                } else {
                    set(feature, pos[feature]);
                }
                p += q;
            }

            cerr << "-------------" << endl;

            if (cmp(ans_value, best_value)) {
                best_value = ans_value;
                best = location;
            }
        }

        auto ans = rect(n_features);
        for (size_t i = 0; i < n_features; i++) {
            ans.set(i, make_pair(best[i], best[i]));
        }
        return make_pair(best_value, ans);  // I divide by forest size in daddy.cpp
    }
};
