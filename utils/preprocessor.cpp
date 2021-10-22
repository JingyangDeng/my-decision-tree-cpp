#include "preprocessor.h"

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

void Preprocessor::train_test_split(const std::vector<std::vector<int>>& data, const std::vector<double>& ratio,
                                    std::vector<std::vector<int>>& train_data, std::vector<int>& train_label,
                                    std::vector<std::vector<int>>& val_data, std::vector<int>& val_label,
                                    std::vector<std::vector<int>>& test_data, std::vector<int>& test_label) {
    int N = data.size();
    int m = data[0].size();
    int test_sz = (int)(N * ratio[2] + 0.5);
    int val_sz = (int)(N * ratio[1] + 0.5);
    int train_sz = N - test_sz - val_sz;

    train_data.resize(train_sz, std::vector<int>(m - 1));
    val_data.resize(val_sz, std::vector<int>(m - 1));
    test_data.resize(test_sz, std::vector<int>(m - 1));
    train_label.resize(train_sz);
    val_label.resize(val_sz);
    test_label.resize(test_sz);

    std::vector<int> idx(N);
    for (int i = 0; i < N; i++) {
        idx[i] = i;
    }
    for (int k = 0; k < N; k++) {
        if (k < test_sz + val_sz) {
            int p = k + rand() % (N - k);
            std::swap(idx[k], idx[p]);
            if (k < test_sz) {
                for (int j = 0; j < m - 1; j++) {
                    test_data[k][j] = data[idx[k]][j];
                }
                test_label[k] = data[idx[k]][m - 1];
            } else {
                for (int j = 0; j < m - 1; j++) {
                    val_data[k - test_sz][j] = data[idx[k]][j];
                }
                val_label[k - test_sz] = data[idx[k]][m - 1];
            }
        } else {
            for (int j = 0; j < m - 1; j++) {
                train_data[k - test_sz - val_sz][j] = data[idx[k]][j];
            }
            train_label[k - test_sz - val_sz] = data[idx[k]][m - 1];
        }
    }
}
