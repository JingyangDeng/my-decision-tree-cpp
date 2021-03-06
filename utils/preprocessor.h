#ifndef _PREPROCESSOR_H
#define _PREPROCESSOR_H

#include <string>
#include <unordered_map>
#include <vector>

class Preprocessor {
public:
    static void factorize(const std::vector<std::vector<std::string>>& data_str, std::vector<std::vector<int>>& data,
                          std::vector<std::vector<std::string>>& dict);

    static void train_test_split(const std::vector<std::vector<int>>& data, const std::vector<double>& ratio,
                                 std::vector<std::vector<int>>& train_data, std::vector<int>& train_label,
                                 std::vector<std::vector<int>>& val_data, std::vector<int>& val_label,
                                 std::vector<std::vector<int>>& test_data, std::vector<int>& test_label);
};

#endif
