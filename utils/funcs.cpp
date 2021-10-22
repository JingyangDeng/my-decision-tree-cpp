#include "funcs.h"

int find(std::unordered_map<int, double>& map, double target) {
    for (auto& it : map) {
        if (it.second == target) {
            return it.first;
        }
    }
    return -1;
}

void assign(Dataset* train_ds, const std::unordered_set<int>& indices, int f,
            std::unordered_map<int, std::unordered_set<int>>& new_indices) {
    const auto& train_data = train_ds->get_data();
    for (int i : indices) {
        new_indices[train_data[i][f]].insert(i);
    }
}

bool label_is_same(Dataset* train_ds, const std::unordered_set<int>& indices){
    const auto& train_label = train_ds->get_label();
    int label = train_label[*indices.begin()];
    for(int i : indices){
        if(train_label[i] != label){
            return false;
        }
    }
    return true;
}

int most_label(Dataset* train_ds, const std::unordered_set<int>& indices){
    std::unordered_map<int, int> cnt;
    const auto& train_label = train_ds->get_label();
    int max = 0;
    for(int i : indices){
        cnt[train_label[i]]++;
        max = fmax(max, cnt[train_label[i]]);
    }
    for(auto& it : cnt){
        if(it.second == max){
            return it.first;
        }
    }
    return -1;
}
