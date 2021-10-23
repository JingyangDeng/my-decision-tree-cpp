#include "model/c4_5.h"
#include "model/cart.h"
#include "utils/loader.h"

int main() {
    std::string path = "./car-dataset/car.data";
    std::vector<double> ratio = {0.6, 0.2, 0.2};
    Loader loader(path, ratio);

    std::cout << "-------------- C4.5 --------------" << std::endl;

    DecisionTreeC4_5 tree_c45;
    tree_c45.train(loader.train_ds, nullptr);
    tree_c45.test(loader.test_ds);

    std::cout << std::endl;
    std::cout << "-------------- CART --------------" << std::endl;

    DecisionTreeCART tree_cart;
    tree_cart.train(loader.train_ds, loader.val_ds);
    tree_cart.test(loader.test_ds);
    return 0;
}
