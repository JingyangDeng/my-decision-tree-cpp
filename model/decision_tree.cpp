#include "decision_tree.h"

#include <iostream>

TreeNode::TreeNode() {
    feat = -1;
    label = -1;
}

void DecisionTree::train(Dataset* train_ds) {
    int N = train_ds->get_data().size();
    int m = train_ds->get_data()[0].size();
    std::unordered_set<int> indices = {};
    std::unordered_set<int> features = {};

    for (int i = 0; i < N; i++) {
        indices.insert(i);
    }
    for (int j = 0; j < m; j++) {
        features.insert(j);
    }

    root = build_tree(train_ds, indices, features);
    prune(train_ds);
}

int DecisionTree::predict(const std::vector<int>& sample) {
    TreeNode* cur = root;
    while (cur->label < 0) {
        cur = cur->child[sample[cur->feat]];
    }
    return cur->label;
}

void DecisionTree::test(Dataset* test_ds) {
    const auto& test_data = test_ds->get_data();
    const auto& test_label = test_ds->get_label();
    int cnt = 0, N = test_label.size();
    for (int i = 0; i < N; i++) {
        int pred = predict(test_data[i]);
        if (pred == test_label[i])
            cnt++;
    }
    std::cout << "accuracy = " << 1. * cnt / N << std::endl;
}
