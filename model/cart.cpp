#include "cart.h"

#include "../utils/funcs.h"
#include "../utils/math.h"

TreeNode* DecisionTreeCART::build_tree(Dataset* train_ds, const std::unordered_set<int>& indices,
                                       const std::unordered_set<int>& features,
                                       std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) {

    TreeNode* root = new TreeNode();

    if (indices.size() <= MIN_LEAF) {
        root->label = most_label(train_ds, indices);
        data_contained[root] = indices;
        return root;
    }

    if (features.empty()) {
        root->label = most_label(train_ds, indices);
        data_contained[root] = indices;
        return root;
    }

    const auto& train_data = train_ds->get_data();
    const auto& train_label = train_ds->get_label();
    std::unordered_map<int, int> lcnt;
    for (int i : indices) {
        lcnt[train_label[i]]++;
    }
    double g = gini(lcnt, indices.size());

    if (g < EPS_GINI) {
        root->label = most_label(train_ds, indices);
        data_contained[root] = indices;
        return root;
    }

    std::unordered_map<int, std::unordered_map<int, double>> gidx = {};
    double gini_min = 1.;
    for (int f : features) {
        std::unordered_set<int> f_value = {};
        for (int i : indices) {
            f_value.insert(train_data[i][f]);
        }
        for (int v : f_value) {
            gidx[f][v] = split_cond_gini(train_ds, indices, f, v);
            gini_min = fmin(gini_min, gidx[f][v]);
        }
    }

    std::pair<int, int> p = find(gidx, gini_min);

    root->feat = p.first;
    root->label = most_label(train_ds, indices);
    data_contained[root] = indices;

    std::unordered_map<int, std::unordered_set<int>> new_indices;
    assign(train_ds, indices, p, new_indices);

    std::unordered_set<int> new_features = features;
    new_features.erase(p.first);

    root->child[p.second] = build_tree(train_ds, new_indices[p.second], new_features, data_contained);

    std::unordered_set<int> new_f_value = {};
    for (int i : new_indices[DEFAULT]) {
        new_f_value.insert(train_data[i][p.first]);
    }

    if (new_f_value.size() > 1) {
        root->child[DEFAULT] = build_tree(train_ds, new_indices[DEFAULT], features, data_contained);
    } else {
        root->child[DEFAULT] = build_tree(train_ds, new_indices[DEFAULT], new_features, data_contained);
    }

    return root;
}

void DecisionTreeCART::prune(Dataset* train_ds, Dataset* val_ds,
                             std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) {
    dfs(root, train_ds, val_ds, data_contained);
}

void DecisionTreeCART::dfs(TreeNode* root, Dataset* train_ds, Dataset* val_ds,
                           std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) {
}
