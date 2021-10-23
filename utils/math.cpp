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

void count(Dataset* train_ds, const std::unordered_set<int>& indices, int f, std::unordered_map<int, std::unordered_map<int, int>>& cnt, std::unordered_map<int, int>& fmap){
    const auto& train_data = train_ds->get_data();
    const auto& train_label = train_ds->get_label();
    for (int i : indices) {
        int v = train_data[i][f];
        int l = train_label[i];
        fmap[v]++;
        cnt[v][l]++;
    }
}

void count(Dataset* train_ds, const std::unordered_set<int>& indices, int f, std::unordered_map<int, std::unordered_map<int, int>>& cnt, std::unordered_map<int, int>& fmap, std::unordered_map<int, int>& lmap){
    const auto& train_data = train_ds->get_data();
    const auto& train_label = train_ds->get_label();
    for (int i : indices) {
        int v = train_data[i][f];
        int l = train_label[i];
        lmap[l]++;
        fmap[v]++;
        cnt[v][l]++;
    }
}

double info_gain_ratio(Dataset* train_ds, const std::unordered_set<int>& indices, int feature) {
    std::unordered_map<int, std::unordered_map<int, int>> cnt;
    std::unordered_map<int, int> fmap;
    std::unordered_map<int, int> lmap;
    count(train_ds, indices, feature, cnt, fmap, lmap);
    double h = entropy(lmap, indices.size());
    double hc = cond_entropy(fmap, cnt, indices.size());
    double hf = entropy(fmap, indices.size());
    return (h - hc) / hf;
}

double loss_entropy(Dataset* train_ds, const std::unordered_set<int>& indices) {
    const auto& train_label = train_ds->get_label();
    int n = indices.size();
    std::unordered_map<int, int> cnt;
    for (int i : indices) {
        cnt[train_label[i]]++;
    }
    return n * entropy(cnt, n);
}

double gini(std::unordered_map<int, int>& pmap, int sum) {
    double ret = 1.;
    for (auto& it : pmap) {
        ret -= pow((double)it.second / sum, 2);
    }
    return ret;
}

double cond_gini(std::unordered_map<int, int>& fmap, std::unordered_map<int, std::unordered_map<int, int>>& cnt,
                 int sum) {
    double gc = 0;
    for (auto& it : fmap) {
        gc += gini(cnt[it.first], it.second) * it.second / sum;
    }
    return gc;
}

double split_cond_gini(Dataset* train_ds, const std::unordered_set<int>& indices, int feature, int value) {
    int D = indices.size();
    const auto& train_data = train_ds->get_data();
    const auto& train_label = train_ds->get_label();
    std::unordered_map<int, std::unordered_map<int, int>> cnt;
    std::unordered_map<int, int> fmap;
    for (int i : indices) {
        int f = train_data[i][feature];
        int l = train_label[i];
        if (f == value) {
            cnt[f][l]++;
            fmap[f]++;
        } else {
            cnt[-1][l]++;
            fmap[-1]++;
        }
    }
    return cond_gini(fmap, cnt, D);
}

double loss_gini(Dataset* train_ds, const std::unordered_set<int>& indices) {
    const auto& train_label = train_ds->get_label();
    int n = indices.size();
    std::unordered_map<int, int> cnt;
    for (int i : indices) {
        cnt[train_label[i]]++;
    }
    return n * gini(cnt, n);
}


int select_split(Dataset* train_ds, const std::unordered_set<int>& indices, int f, std::unordered_set<int>& selected_values){
    selected_values.clear();
    std::unordered_map<int, std::unordered_map<int, int>> cnt;
    std::unordered_map<int, int> fmap;
    count(train_ds, indices, f, cnt, fmap); 
    double wavg_entropy = cond_entropy(fmap, cnt, indices.size());
    for (auto& it : fmap) {
        double h = entropy(cnt[it.first], it.second);
        if(h <= wavg_entropy){
            selected_values.insert(it.first);
        }
    }
    return fmap.size();
}