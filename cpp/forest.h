#pragma once

#include <cassert>
#include <iostream>
#include <cassert>

#include "util.h"

struct node {
    size_t index, size, depth;
    size_t feature;
    node *left, *right;
    double threshold, value, shit;
    size_t left_index, right_index;
    size_t n_features;
    vector<leaf> leaves;
    rect area;

    node(size_t index, size_t feature, double threshold, double value, size_t left_index, size_t right_index,
         size_t n_features)
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

    void precalc(const comp& cmp) {
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
        merge(left->leaves.begin(), left->leaves.end(), right->leaves.begin(), right->leaves.end(),
              back_inserter(leaves), cmp);

        value = cmp.best(left->value, right->value);
        shit = cmp.worst(left->shit, right->shit);
        size = left->size + right->size;
        depth = max(left->depth, right->depth) + 1;
    }

    bool is_leaf() const {
        return left == nullptr;
    }

    void print(FILE *stream, size_t depth = 0) const {
        for (size_t i = 0; i < depth; i++) {
            fprintf(stream, "|\t");
        }
        if (is_leaf()) {
            fprintf(stream, "(%lf)\n", value);
        } else {
            fprintf(stream, "[%zu %lf]\n", feature, threshold);
            left->print(stream, depth + 1);
            right->print(stream, depth + 1);
        }
    }

    node* predict(const vector<double>& value) {
        assert(value.size() == n_features);
        if (is_leaf()) return this;
        if (value[feature] > threshold) {
            return right->predict(value);
        } else {
            return left->predict(value);
        }
    }
};

class forest {
protected:
    size_t forest_size;
    size_t n_features;
    comp cmp;
    rect limits;

    vector<node*> roots;
    vector<node*> trees;

    size_t iterations;

public:
    forest(size_t forest_size, size_t n_features, const comp& cmp)
        : forest_size(forest_size), n_features(n_features), cmp(cmp), limits(n_features) {
    }

    void read_nodes(FILE* stream) {
        vector<node*> nodes;
        for (size_t i = 0; i < forest_size; i++) {
            int size;
            fscanf(stream, "%d", &size);
            nodes.reserve(size);
            for (int j = 0; j < size; j++) {
                int index, feature, left_index, right_index;
                double threshold, value;
                fscanf(stream, "%d %d %lf %lf %d %d", &index, &feature, &threshold, &value, &left_index, &right_index);
                nodes[index] = new node(index, feature, threshold, value, left_index, right_index, n_features);
            }
            for (int j = 0; j < size; j++) {
                if (nodes[j]->left_index != (size_t)-1) {
                    nodes[j]->left = nodes[nodes[j]->left_index];
                    nodes[j]->right = nodes[nodes[j]->right_index];
                }
            }
            nodes[0]->precalc(cmp);
            trees.push_back(nodes[0]);
        }
        roots = vector<node*>(trees);

        // reading limits
        for (size_t i = 0; i < n_features; i++) {
            double min, max;
            fscanf(stream, "%lf %lf\n", &min, &max);
            limits.set(i, make_pair(min, max));
        }

        //reading params
        fscanf(stream, "%zu\n", &iterations);
    }

    double predict(const vector<double>& value) {
        double ans = 0;
        for (auto root : trees) {
            ans += root->predict(value)->value;
        }
        return ans;
    }

    virtual pair<double, rect> inverse() = 0;
};
