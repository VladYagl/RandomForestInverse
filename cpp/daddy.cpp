#include <ctime>

#include "util.h"
#include "heuristic.h"
#include "basic.h"
#include "random.h"
#include "gena.h"

int main() {
    /* freopen("../test_forest.txt", "r", stdin); */

    char type[8];
    char algo[20];
    double error;
    size_t n_features, forest_size;
    scanf("%s\n", algo);
    scanf("%s\n", type);
    scanf("%lf\n", &error);
    scanf("%zu %zu", &n_features, &forest_size);
    fprintf(stderr, "type = [%s]\n", type);
    fprintf(stderr, "forest_size = [%zu]\n", forest_size);
    fprintf(stderr, "algorithm = [%s]\n", algo);

    forest* gump = nullptr;
    if (strcmp(algo, "basic") == 0) {
        gump = new basic(forest_size, n_features, std::move(comp(type, error)));
    } else if (strcmp(algo, "heuristic") == 0) {
        gump = new heuristic(forest_size, n_features, std::move(comp(type, error)));
    } else if (strcmp(algo, "random") == 0) {
        gump = new rnd(forest_size, n_features, std::move(comp(type, error)));
    } else if (strcmp(algo, "gena") == 0) {
        gump = new gena(forest_size, n_features, std::move(comp(type, error)));
    } else {
        fprintf(stderr, "WRONG ALGO\n");
    }
    
    gump->read_nodes(stdin);
    fprintf(stderr, "forest reading completed!\n");
    clock_t start_time = clock();
    auto result = gump->inverse();
    double elapsed_time = double(clock() - start_time) / CLOCKS_PER_SEC;
    double ans_value = result.first;
    rect ans = result.second;

    printf("%lf\n", elapsed_time);
    printf("%lf\n", ans_value / forest_size);
    ans.print(stdout);

    return 0;
}

