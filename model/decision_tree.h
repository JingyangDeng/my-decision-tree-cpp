#ifndef _DECISION_TREE_H
#define _DECISION_TREE_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model.h"

class TreeNode {
public:
    int feat;
    int label;
    std::unordered_map<int, TreeNode*> child;
    TreeNode();
};

class DecisionTree : public Model {
protected:
    TreeNode* root;
    virtual TreeNode* build_tree(Dataset* train_ds, std::unordered_set<int>& indices,
                                 std::unordered_set<int>& features) = 0;
    virtual void prune(Dataset* train_ds) = 0;
    int predict(const std::vector<int>& sample);

public:
    void train(Dataset* train_ds);
    void test(Dataset* test_ds);
};

#endif
