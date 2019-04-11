#pragma once

#include <cassert>
#include <iostream>

#include "util.h"

class forest {
    size_t forest_size;
    size_t n_features;

    vector<node*> roots;
    vector<node*> trees;
    vector<size_t> leaves;
    comp cmp;

    rect current;
    rect ans;
    double ans_value;
    size_t ans_count;
    size_t before_split_count = 0;
    size_t split_count = 0;
    size_t bad_count = 0;
    size_t sync_count = 0;

    inline pair<double, const rect*> leaf(size_t i) const {
        return roots[i]->leaves[leaves[i]];
    }

    void sync_values() {
        double possible = 0;
        double est = 0;
        bool all_leaves = true;
        sync_count++;
        auto old_leaves = leaves;
        auto old_trees = trees;

        // make all intersect current
        for (size_t i = 0; i < trees.size(); i++) {
            // v TODO: epsilon or some shit
            while (!trees[i]->is_leaf() && trees[i]->threshold < current.lower[trees[i]->feature] + 1e-8) {
                trees[i] = trees[i]->right;
            }
            while (!trees[i]->is_leaf() && trees[i]->threshold > current.upper[trees[i]->feature] - 1e-8) {  // >=
                trees[i] = trees[i]->left;
            }
        }

        // update possible leaves
        for (size_t i = 0; i < trees.size(); i++) {
            node* root = trees[i];
            if (!root->is_leaf()) {
                if (!intersects(*leaf(i).second, current)) {
                    while (!intersects(*leaf(i).second, current)) {
                        leaves[i]++;
                        if (leaves[i] >= roots[i]->leaves.size()) {
                            leaves = std::move(old_leaves);
                            trees = std::move(old_trees);
                            return;
                        }
                    }
                }
                all_leaves = false;
            }
            possible += root->value;
            est += leaf(i).first;
            /* est = possible; */
        }

        // update ans if all leafs
        if (all_leaves) {
            if (possible == ans_value) {
                ans_count++;
            }
            if (cmp(possible, ans_value)) {
                ans_value = possible;
                ans_count = 1;
                ans.copy(current);
                cerr << ans_value << "\t" << sync_count << endl;
            }
            leaves = std::move(old_leaves);
            trees = std::move(old_trees);
            return;
        }

        before_split_count++;
        if (cmp(est, ans_value)) {
            split();
        }

        leaves = std::move(old_leaves);
        trees = std::move(old_trees);
    }

    void split() {
        size_t max_pos = -1;
        double max_diff = -1;
        double possible = 0;
        for (size_t i = 0; i < trees.size(); i++) {
            node* root = trees[i];
            possible += root->value;
            if (!root->is_leaf()) {
                double diff = abs(root->left->value - root->right->value);
                if (diff > max_diff) {
                    max_diff = diff;
                    max_pos = i;
                }
            }
        }

        node* split_root = trees[max_pos];
        node* best = split_root->left;
        node* worst = split_root->right;
        auto old_border = make_pair(current.lower[split_root->feature], current.upper[split_root->feature]);
        auto best_border = make_pair(old_border.first, split_root->threshold);
        auto worst_border = make_pair(split_root->threshold, old_border.second);

        if (!cmp(split_root->left->value, split_root->right->value)) {
            swap(best, worst);
            swap(best_border, worst_border);
        }

        trees[max_pos] = best;
        current.set(split_root->feature, best_border);
        sync_values();

        split_count++;
        if (cmp(possible - best->value + worst->value, ans_value)) {
            trees[max_pos] = worst;
            current.set(split_root->feature, worst_border);
            sync_values();
            bad_count++;
        }

        trees[max_pos] = split_root;
        current.set(split_root->feature, old_border);
    }

public:
    forest(size_t forest_size, size_t n_features, char* type)
        : forest_size(forest_size), n_features(n_features), leaves(forest_size), cmp(type) {
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
    }

    pair<double, rect> inverse() {
        double sum = 0;
        size_t size = 0;
        size_t max_depth = 0;
        size_t depth_sum = 0;
        for (auto root : trees) {
            sum += root->shit;
            size += root->size;
            max_depth = max(max_depth, root->depth);
            depth_sum = depth_sum + root->depth;
        }
        fprintf(stderr, "Max depth = %zu, Avg. depth = %zu\n", max_depth, depth_sum / forest_size);

        current = rect(n_features);
        ans = rect(n_features);
        ans_value = sum;
        sync_values();

        fprintf(stderr, "Done\nHeuristic effectiveness: %.0lf%% (%zu / %zu)\n",
                (double)(split_count - bad_count) / split_count * 100, split_count - bad_count, split_count);
        fprintf(stderr, "Optimization effectiveness: %.0lf%% (%zu / %zu)\n",
                (double)(split_count) / before_split_count * 100, split_count, before_split_count);

        /* fprintf(stderr, "Number of zones: %zu\n", ans_count);; */
        /* assert(ans_count == 1); */

        return make_pair(ans_value, ans);
    }
};
