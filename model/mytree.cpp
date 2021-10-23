#include "mytree.h"

#include "../utils/funcs.h"
#include "../utils/math.h"

TreeNode* MyDecisionTree::build_tree(Dataset* train_ds, const std::unordered_set<int>& indices,
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

    if (gain_max < EPS_INR_MYTREE) {
        root->label = most_label(train_ds, indices);
        data_contained[root] = indices;
        return root;
    }

    int f = find(gain_ratio, gain_max);
    root->feat = f;

    root->label = most_label(train_ds, indices);
    data_contained[root] = indices;

    std::unordered_set<int> selected_values;
    int value_num = select_split(train_ds, indices, f, selected_values);

    std::unordered_map<int, std::unordered_set<int>> new_indices;
    assign(train_ds, indices, f, selected_values, new_indices);

    std::unordered_set<int> new_features = features;
    new_features.erase(f);

    for (auto& it : new_indices) {
        if (it.first == DEFAULT && selected_values.size() < value_num - 1) {
            root->child[DEFAULT] = build_tree(train_ds, it.second, features, data_contained);
        } else {
            root->child[it.first] = build_tree(train_ds, it.second, new_features, data_contained);
        }
    }

    return root;
}

// same as the pruning alogrithm of c4.5
void MyDecisionTree::prune(Dataset* train_ds, Dataset* val_ds,
                           std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) {
    dfs(this->root, train_ds, val_ds, data_contained);
}

void MyDecisionTree::dfs(TreeNode* root, Dataset* train_ds, Dataset* val_ds,
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
        double ca = loss_entropy(train_ds, data_contained[root]) + ALPHA_MYTREE;
        double cb = 0;
        for (auto& it : root->child) {
            cb += loss_entropy(train_ds, data_contained[it.second]) + ALPHA_MYTREE;
        }
        if (ca <= cb) {
            root->feat = -1;
            return;
        }
    }
}
