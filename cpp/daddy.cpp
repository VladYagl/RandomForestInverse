#include "util.h"
#include "forest.h"

int main() {
    /* freopen("test_forest.txt", "r", stdin); */

    char type[8];
    size_t n_features, forest_size;
    scanf("%s\n", type);
    scanf("%zu %zu", &n_features, &forest_size);
    fprintf(stderr, "type = [%s]\n", type);
    fprintf(stderr, "forest_size = [%zu]\n", forest_size);

    forest gump(forest_size, n_features, type);
    gump.read_nodes(stdin);
    auto result = gump.inverse();
    double ans_value = result.first;
    rect ans = result.second;

    printf("%lf\n", ans_value / forest_size);
    ans.print(stdout);

    return 0;
}

