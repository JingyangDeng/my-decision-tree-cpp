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
    const auto& val_data = val_ds->get_data();
    const auto& val_label = val_ds->get_label();

    std::vector<double> acc;
    int val_sz = val_label.size();
    int cnt = 0;
    for (int i = 0; i < val_sz; i++) {
        int c = predict(val_data[i]);
        cnt += (c == val_label[i]) ? 1 : 0;
    }
    acc.emplace_back((double)cnt / val_sz);

    while (root->feat >= 0) {
        std::vector<std::pair<double, TreeNode*>> g;
        dfs(root, train_ds, data_contained, g);
        int k = std::min_element(g.begin(), g.end()) - g.begin();
        TreeNode* node = g[k].second;
        alpha.emplace_back(g[k].first);
        pruned_node.emplace_back(node);
        pruned_feat.emplace_back(node->feat);
        node->feat = -1;

        int cnt = 0;
        for (int i = 0; i < val_sz; i++) {
            int c = predict(val_data[i]);
            cnt += (c == val_label[i]) ? 1 : 0;
        }
        acc.emplace_back((double)cnt / val_sz);
    }
    int k = std::max_element(acc.begin(), acc.end()) - acc.begin();
    for (int i = k; i < (int)pruned_node.size(); i++) {
        pruned_node[i]->feat = pruned_feat[i];
    }
}

std::pair<int, double> DecisionTreeCART::dfs(TreeNode* root, Dataset* train_ds,
                                             std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained,
                                             std::vector<std::pair<double, TreeNode*>>& g) {
    if (root->feat == -1) {
        return std::make_pair(1, loss_gini(train_ds, data_contained[root]));
    }

    int num_leaf = 0;
    double loss_leaf = 0;
    for (auto& it : root->child) {
        auto p = dfs(it.second, train_ds, data_contained, g);
        num_leaf += p.first;
        loss_leaf += p.second;
    }
    double loss = loss_gini(train_ds, data_contained[root]);
    double gt = (loss - loss_leaf) / (num_leaf - 1);
    g.emplace_back(std::make_pair(gt, root));
    return std::make_pair(num_leaf, loss_leaf);
}
