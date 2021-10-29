#ifndef _LOADER_H
#define _LOADER_H

#include <fstream>
#include <iostream>
#include <string>

#include "dataset.h"

class Loader {
private:
    std::string path;
    void load(const std::vector<double>& ratio);
    void load_data(const std::string& path, Dataset* ds);
    void save_data(const std::string& path, Dataset* ds);
    std::vector<std::vector<std::string>> dict;

public:
    Dataset* train_ds;
    Dataset* test_ds;
    Dataset* val_ds;
    Loader(std::string path, const std::vector<double>& test_ratio);
    const std::vector<std::vector<std::string>>& get_dict();
    void show_dict();
    ~Loader();
};

#endif
