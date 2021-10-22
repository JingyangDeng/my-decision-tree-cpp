#ifndef _DECISION_TREE_H
#define _DECISION_TREE_H

#include <iostream>
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
    TreeNode* root = nullptr;
    virtual TreeNode* build_tree(Dataset* train_ds, const std::unordered_set<int>& indices,
                                 const std::unordered_set<int>& features,
                                 std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) = 0;

    virtual void prune(Dataset* train_ds, std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) = 0;
    int predict(const std::vector<int>& sample);

public:
    void train(Dataset* train_ds);
    void test(Dataset* test_ds);
};

#endif
