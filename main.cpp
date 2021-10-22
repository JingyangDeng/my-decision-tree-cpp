#include "utils/loader.h"
#include "model/c4_5.h"

int main() {
    std::string path = "./car-dataset/car.data";
    std::vector<double> ratio = {0.6, 0.2, 0.2};
    Loader loader(path, ratio);
    DecisionTreeC4_5 tree;

    tree.train(loader.train_ds);
    tree.test(loader.test_ds);
    return 0;
}
