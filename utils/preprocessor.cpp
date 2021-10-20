#include "preprocessor.h"

#include <iostream>
#include <unordered_map>

void Preprocessor::factorize(const std::vector<std::vector<std::string>>& data_str, std::vector<std::vector<int>>& data,
                             std::vector<std::vector<std::string>>& dict) {
    int N = data_str.size();
    int m = data_str[0].size();
    data.resize(N, std::vector<int>(m));
    dict.resize(m);
    for (int j = 0; j < m; j++) {
        std::unordered_map<std::string, int> map = {};
        int idx = 0;
        for (int i = 0; i < N; i++) {
            if (map.find(data_str[i][j]) == map.end()) {
                map[data_str[i][j]] = idx++;
                dict[j].emplace_back(data_str[i][j]);
            }
            data[i][j] = map[data_str[i][j]];
        }
    }
}

void Preprocessor::train_test_split(const std::vector<std::vector<int>>& data, double test_ratio,
                                    std::vector<std::vector<int>>& train_data, std::vector<int>& train_label,
                                    std::vector<std::vector<int>>& test_data, std::vector<int>& test_label) {
    int N = data.size();
    int n = (int)(N * test_ratio + 0.5);
    int m = data[0].size();

    train_data.resize(N - n, std::vector<int>(m - 1));
    test_data.resize(n, std::vector<int>(m - 1));
    train_label.resize(N - n);
    test_label.resize(n);

    std::vector<int> idx(N);
    for (int i = 0; i < N; i++) {
        idx[i] = i;
    }
    for (int k = 0; k < N; k++) {
        if (k < n) {
            int p = k + rand() % (N - k);
            std::swap(idx[k], idx[p]);
            for (int j = 0; j < m - 1; j++) {
                test_data[k][j] = data[idx[k]][j];
            }
            test_label[k] = data[idx[k]][m - 1];
        } else {
            for (int j = 0; j < m - 1; j++) {
                train_data[k - n][j] = data[idx[k]][j];
            }
            train_label[k - n] = data[idx[k]][m - 1];
        }
    }
}
