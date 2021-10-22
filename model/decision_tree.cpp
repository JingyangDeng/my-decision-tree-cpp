#include "decision_tree.h"

TreeNode::TreeNode() {
    feat = -1;
    label = -1;
}

void DecisionTree::train(Dataset* train_ds) {
    std::cout << "training..." << std::endl;
    int N = train_ds->get_data().size();
    int m = train_ds->get_data()[0].size();
    std::unordered_set<int> indices = {};
    std::unordered_set<int> features = {};
    std::unordered_map<TreeNode*, std::unordered_set<int>> data_contained = {};

    for (int i = 0; i < N; i++) {
        indices.insert(i);
    }
    for (int j = 0; j < m; j++) {
        features.insert(j);
    }

    root = build_tree(train_ds, indices, features, data_contained);
    prune(train_ds, data_contained);
}

int DecisionTree::predict(const std::vector<int>& sample) {
    TreeNode* cur = root;
    while (cur->feat >= 0) {
        if (cur->child.find(sample[cur->feat]) == cur->child.end()) {
            return cur->label;
        }
        cur = cur->child[sample[cur->feat]];
    }
    return cur->label;
}

void DecisionTree::test(Dataset* test_ds) {
    std::cout << "testing..." << std::endl;
    const auto& test_data = test_ds->get_data();
    const auto& test_label = test_ds->get_label();
    int cnt = 0, N = test_label.size();
    for (int i = 0; i < N; i++) {
        int pred = predict(test_data[i]);
        std::cout << "pred: " << pred << " truth: " << test_label[i] << std::endl;
        if (pred == test_label[i])
            cnt++;
    }
    std::cout << "accuracy = " << 1. * cnt / N << std::endl;
}
