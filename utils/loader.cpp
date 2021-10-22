#include "loader.h"

#include "csv_reader.h"
#include "preprocessor.h"

Loader::Loader(std::string path, const std::vector<double>& ratio) {
    this->path = path;
    train_ds = new Dataset();
    val_ds = new Dataset();
    test_ds = new Dataset();
    load(ratio);
}

inline bool is_file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

void Loader::load_data(const std::string& path, Dataset* ds) {
    std::ifstream f(path);
    std::string line_str = {};
    while (getline(f, line_str)) {
        std::stringstream ss(line_str);
        std::string str;
        std::vector<int> sample = {};
        while (getline(ss, str, ' '))
            sample.emplace_back(stoi(str));
        ds->label.emplace_back(sample.back());
        sample.pop_back();
        ds->data.emplace_back(sample);
    }
}

void Loader::load(const std::vector<double>& ratio) {
    if (is_file_exists(path + ".train") && is_file_exists(path + ".val") && is_file_exists(path + ".test")) {
        load_data(path + ".train", train_ds);
        load_data(path + ".val", val_ds);
        load_data(path + ".test", test_ds);
        return;
    }
    CSVReader reader(path);
    std::vector<std::vector<int>> data;
    Preprocessor::factorize(reader.get_data(), data, dict);
    Preprocessor::train_test_split(data, ratio, train_ds->data, train_ds->label, val_ds->data, val_ds->label,
                                   test_ds->data, test_ds->label);
}

const std::vector<std::vector<std::string>>& Loader::get_dict() {
    return dict;
}

void Loader::show_dict() {
    for (int i = 0; i < (int)dict.size(); i++) {
        for (int j = 0; j < (int)dict[i].size(); j++) {
            std::cout << dict[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Loader::~Loader() {
    delete train_ds;
    delete test_ds;
    delete val_ds;
}
