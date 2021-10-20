#include <string>

#include "utils/csv_reader.h"
#include "utils/dataset.h"

int main() {
    std::string path = "./car-dataset/car.data";
    Dataset* d = new Dataset(path, 0.3);
    d->show();
    return 0;
}
