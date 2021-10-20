#ifndef _LOADER_H
#define _LOADER_H

#include <string>

#include "dataset.h"

class Loader {
private:
    std::string path;
    void load(double test_ratio);
    std::vector<std::vector<std::string>> dict;

public:
    Dataset* train_ds;
    Dataset* test_ds;
    Loader(std::string path, double test_ratio);
    const std::vector<std::vector<std::string>>& get_dict();
    ~Loader();
};

#endif
