#include "math.h"

double plogp(double p) {
    if (p == 0)
        return 0;
    return -p * log2(p);
}

double entropy(std::unordered_map<int, int>& pmap, int sum) {
    double ret = 0;
    for (auto& it : pmap) {
        ret += plogp((double)it.second / sum);
    }
    return ret;
}

double cond_entropy(std::unordered_map<int, int>& fmap, std::unordered_map<int, std::unordered_map<int, int>>& cnt,
                    int sum) {
    double hc = 0;
    for (auto& it : fmap) {
        hc += entropy(cnt[it.first], it.second) * it.second / sum;
    }
    return hc;
}

double info_gain_ratio(Dataset* train_ds, const std::unordered_set<int>& indices, int feature) {
    const auto& train_data = train_ds->get_data();
    const auto& train_label = train_ds->get_label();
    int D = indices.size();

    std::unordered_map<int, std::unordered_map<int, int>> cnt;
    std::unordered_map<int, int> fmap;
    std::unordered_map<int, int> lmap;
    for (int i : indices) {
        int f = train_data[i][feature];
        int l = train_label[i];
        fmap[f]++;
        lmap[l]++;
        cnt[f][l]++;
    }
    double h = entropy(lmap, D);
    double hc = cond_entropy(fmap, cnt, D);
    double hf = entropy(fmap, D);
    return (h - hc) / hf;
}

double loss(Dataset* train_ds, const std::unordered_set<int>& indices) {
    const auto& train_label = train_ds->get_label();
    int n = indices.size();
    std::unordered_map<int, int> cnt;
    for (int i : indices) {
        cnt[train_label[i]]++;
    }
    return n * entropy(cnt, n);
}
