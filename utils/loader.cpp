
#include "loader.h"

#include "csv_reader.h"
#include "preprocessor.h"

Loader::Loader(std::string path, double test_ratio) {
    this->path = path;
    train_ds = new Dataset();
    test_ds = new Dataset();
    load(test_ratio);
}

void Loader::load(double test_ratio) {
    CSVReader reader(path);
    std::vector<std::vector<int>> data;
    Preprocessor::factorize(reader.get_data(), data, dict);
    Preprocessor::train_test_split(data, test_ratio, train_ds->data, train_ds->label, test_ds->data, test_ds->label);
}

const std::vector<std::vector<std::string>>& Loader::get_dict() {
    return dict;
}

Loader::~Loader() {
    delete train_ds;
    delete test_ds;
}
