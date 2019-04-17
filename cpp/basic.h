#pragma once

#include "forest.h"

class basic : public forest {
    rect current;
    rect ans;

    double ans_value;

    size_t sync_count = 0;

    void sync_values() {
        double possible = 0;
        sync_count++;
        bool all_leaves = true;
        auto old_trees = trees;

        // make all intersect current
        for (size_t i = 0; i < trees.size(); i++) {
            bool repeat = true;
            while (repeat) {
                repeat = false;
                while (!trees[i]->is_leaf() && trees[i]->threshold < current.lower[trees[i]->feature] + 1e-8) {
                    trees[i] = trees[i]->right;
                    repeat = true;
                }
                while (!trees[i]->is_leaf() && trees[i]->threshold > current.upper[trees[i]->feature] - 1e-8) {  // >=
                    trees[i] = trees[i]->left;
                    repeat = true;
                }
            }
            node* root = trees[i];
            if (!root->is_leaf()) {
                all_leaves = false;
            }
            possible += root->value;
        }

        // update ans if all leafs
        if (all_leaves) {
            if (cmp(possible, ans_value)) {
                ans_value = possible;
                ans.copy(current);
                cerr << ans_value << "\t" << sync_count << endl;
            }
            trees = std::move(old_trees);
            return;
        }

        if (cmp(possible, cmp.inc(ans_value))) {
            split();
        }

        trees = std::move(old_trees);
    }

    void split() {
        size_t max_pos = -1;
        for (size_t i = 0; i < trees.size(); i++) {
            if (!trees[i]->is_leaf()) {
                max_pos = i;
            }
        }

        max_pos = rand() % trees.size();
        node* split_root = trees[max_pos];
        while (split_root->is_leaf()) {
            max_pos = rand() % trees.size();
            split_root = trees[max_pos];
        }

        /* node* split_root = trees[max_pos]; */
        node* best = split_root->left;
        node* worst = split_root->right;
        auto old_border = make_pair(current.lower[split_root->feature], current.upper[split_root->feature]);
        auto best_border = make_pair(old_border.first, split_root->threshold);
        auto worst_border = make_pair(split_root->threshold, old_border.second);

        trees[max_pos] = best;
        current.set(split_root->feature, best_border);
        sync_values();

        trees[max_pos] = worst;
        current.set(split_root->feature, worst_border);
        sync_values();

        trees[max_pos] = split_root;
        current.set(split_root->feature, old_border);
    }

public:
    basic(size_t forest_size, size_t n_features, const comp& cmp) : forest(forest_size, n_features, cmp) {
    }

    pair<double, rect> inverse() {
        double sum = 0;
        for (auto root : trees) {
            sum += root->shit;
        }

        current = rect(n_features);
        ans = rect(n_features);
        ans_value = sum;
        sync_values();

        return make_pair(ans_value, ans);  // I divide by forest size in daddy.cpp
    }
};
