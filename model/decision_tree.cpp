#include "decision_tree.h"

TreeNode::TreeNode() {
    feat = -1;
    label = -1;
}

// the training process contains two stage: building tree and pruning.
void DecisionTree::train(Dataset* train_ds, Dataset* val_ds) {
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
    prune(train_ds, val_ds, data_contained);
}

// predict the label of a single sample.
int DecisionTree::predict(const std::vector<int>& sample) {
    TreeNode* cur = root;
    while (cur->feat >= 0) {
        //check if value of considered feature is in child.
        if (cur->child.find(sample[cur->feat]) == cur->child.end()) {
            // if not, try default value -1.
            if (cur->child.find(DEFAULT) == cur->child.end()) {
                // if key -1 does not exist neither, return the most common label.
                return cur->label;
            } else {
                cur = cur->child[DEFAULT];
            }
        } else {
            cur = cur->child[sample[cur->feat]];
        }
    }
    return cur->label;
}

// traverse all testing samples and compare their predicted label & ground truth.
void DecisionTree::test(Dataset* test_ds) {
    std::cout << "testing..." << std::endl;
    const auto& test_data = test_ds->get_data();
    const auto& test_label = test_ds->get_label();
    int cnt = 0, N = test_label.size();
    for (int i = 0; i < N; i++) {
        int pred = predict(test_data[i]);
        // std::cout << "pred: " << pred << " truth: " << test_label[i] << std::endl;
        if (pred == test_label[i])
            cnt++;
    }
    std::cout << "accuracy = " << 100. * cnt / N << "\%" << std::endl;
}
