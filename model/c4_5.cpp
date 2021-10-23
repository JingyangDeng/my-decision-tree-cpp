#include "c4_5.h"

#include "../utils/funcs.h"
#include "../utils/math.h"

TreeNode* DecisionTreeC4_5::build_tree(Dataset* train_ds, const std::unordered_set<int>& indices,
                                       const std::unordered_set<int>& features,
                                       std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) {
    TreeNode* root = new TreeNode();

    if (label_is_same(train_ds, indices)) {
        const auto& train_label = train_ds->get_label();
        root->label = train_label[*indices.begin()];
        data_contained[root] = indices;
        return root;
    }

    if (features.empty()) {
        root->label = most_label(train_ds, indices);
        data_contained[root] = indices;
        return root;
    }

    std::unordered_map<int, double> gain_ratio;
    double gain_max = 0;
    for (int f : features) {
        gain_ratio[f] = info_gain_ratio(train_ds, indices, f);
        gain_max = fmax(gain_max, gain_ratio[f]);
    }

    if (gain_max < EPS_INR) {
        root->label = most_label(train_ds, indices);
        data_contained[root] = indices;
        return root;
    }

    int f = find(gain_ratio, gain_max);
    root->feat = f;

    root->label = most_label(train_ds, indices);
    data_contained[root] = indices;

    std::unordered_map<int, std::unordered_set<int>> new_indices;
    assign(train_ds, indices, f, new_indices);

    std::unordered_set<int> new_features = features;
    new_features.erase(f);

    for (auto& it : new_indices) {
        root->child[it.first] = build_tree(train_ds, it.second, new_features, data_contained);
    }

    return root;
}

void DecisionTreeC4_5::prune(Dataset* train_ds, Dataset* val_ds,
                             std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) {
    dfs(this->root, train_ds, val_ds, data_contained);
}

void DecisionTreeC4_5::dfs(TreeNode* root, Dataset* train_ds, Dataset* val_ds,
                           std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) {
    if (root->feat == -1)
        return;

    for (auto& it : root->child) {
        dfs(it.second, train_ds, val_ds, data_contained);
    }

    bool is_last_parent = true;
    for (auto& it : root->child) {
        if (it.second->feat >= 0) {
            is_last_parent = false;
            break;
        }
    }

    if (is_last_parent) {
        double ca = loss(train_ds, data_contained[root]) + ALPHA;
        double cb = 0;
        for (auto& it : root->child) {
            cb += loss(train_ds, data_contained[it.second]) + ALPHA;
        }
        if (ca <= cb) {
            root->feat = -1;
            return;
        }
    }
}
