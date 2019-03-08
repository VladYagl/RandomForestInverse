#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "rect.h"

using namespace std;

size_t forest_size;
size_t n_features;
char type[8];
bool better(double a, double b) {
    if (strcmp(type, "min") == 0)
        return a < b;
    else
        return a > b;
}

double best(double a, double b) {
    if (better(a, b))
        return a;
    else
        return b;
}

double worst(double a, double b) {
    if (better(a, b))
        return b;
    else
        return a;
}

struct node {
    size_t index, size;
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
    }

    void precalc() {
        if (is_leaf()) return;
        left->precalc();
        right->precalc();
        value = best(left->value, right->value);
        shit = worst(left->shit, right->shit);
        size = left->size + right->size;
    }

    bool is_leaf() {
        return left == nullptr;
    }
};

vector<node*> nodes;
vector<node*> trees;
rect current;
rect ans;
double ans_value;
size_t split_count = 0;
size_t bad_count  = 0;

void solve() {
    size_t max_pos = -1;
    double max_diff = -1;
    double possible = 0;
    bool all_leafs = true;
    for (size_t i = 0; i < trees.size(); i++) {
        node* root = trees[i];
        possible += root->value;
        if (!root->is_leaf()) {
            all_leafs = false;
            double diff = abs(root->left->value - root->right->value);
            if (diff > max_diff) {
                max_diff = diff;
                max_pos = i;
            }
        }
    }

    // make all intersect current
    for (size_t i = 0; i < trees.size(); i++) {
        node* root = trees[i];
        if (!root->is_leaf()) {
            if (root->threshold < current.lower[root->feature] + 1e-8) { // TODO: epsilon or some shit
                trees[i] = root->right;
                solve();
                trees[i] = root;
                return;
            }
            if (root->threshold > current.upper[root->feature] - 1e-8) {
                trees[i] = root->left;
                solve();
                trees[i] = root;
                return;
            }
        }
    }

    if (all_leafs) {
        if (better(possible, ans_value)) {
            ans_value = possible;
            ans.copy(current);
        }
        return;
    }

    node* split_root = trees[max_pos];
    node* best = split_root->left;
    node* worst = split_root->right;
    auto old_border = make_pair(current.lower[split_root->feature], current.upper[split_root->feature]);
    auto best_border = make_pair(old_border.first, split_root->threshold);
    auto worst_border = make_pair(split_root->threshold, old_border.second);

    if (better(split_root->left->value, split_root->right->value)) {
        swap(best, worst);
        swap(best_border, worst_border);
    }

    trees[max_pos] = best;
    current.set(split_root->feature, best_border);
    solve();

    split_count++;
    if (better(possible - best->value + worst->value, ans_value)) {
        trees[max_pos] = worst;
        current.set(split_root->feature, worst_border);
        solve();
        bad_count++;
    } 

    trees[max_pos] = split_root;
    current.set(split_root->feature, old_border);
}

int main() {
    /* freopen("test_forest.txt", "r", stdin); */

    scanf("%s\n", type);
    scanf("%zu %zu", &n_features, &forest_size);
    fprintf(stderr, "type = [%s]\n", type);
    fprintf(stderr, "forest_size = [%zu]\n", forest_size);

    for (size_t i = 0; i < forest_size; i++) {
        int size;
        scanf("%d", &size);
        nodes.reserve(size);
        for (int j = 0; j < size; j++) {
            int index, feature, left_index, right_index;
            double threshold, value;
            scanf("%d %d %lf %lf %d %d", &index, &feature, &threshold, &value, &left_index, &right_index);
            nodes[index] = new node(index, feature, threshold, value, left_index, right_index);
        }
        for (int j = 0; j < size; j++) {
            if (nodes[j]->left_index != -1) {
                nodes[j]->left = nodes[nodes[j]->left_index];
                nodes[j]->right = nodes[nodes[j]->right_index];
            }
        }
        nodes[0]->precalc();
        trees.push_back(nodes[0]);
    }

    double sum = 0;
    for (auto root : trees) {
        sum += root->shit;
    }
    current = rect(n_features);
    ans = rect(n_features);
    ans_value = sum;
    solve();

    fprintf(stderr, "Done\nHeuristic effectiveness: %lf%%", (double)(split_count - bad_count) / split_count * 100);

    printf("%lf\n", ans_value / forest_size);
    ans.print(stdout);

    return 0;
}

