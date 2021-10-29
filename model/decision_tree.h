#ifndef _DECISION_TREE_H
#define _DECISION_TREE_H

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model.h"

#define DEFAULT -1

class TreeNode {
public:
    int feat;                                 // the id of feature that current node considers
    int label;                                // the label of instances belonging to current node
    std::unordered_map<int, TreeNode*> child; // map: the value of feat -> next node, set DEFAULT = -1 as default value. 
    TreeNode();
};

// DecisionTree is an abstract class. Its pure virtual methods, build_tree & prune, should be implemented by its subclass. 
class DecisionTree : public Model {
protected:
    TreeNode* root = nullptr;
    virtual TreeNode* build_tree(Dataset* train_ds, const std::unordered_set<int>& indices,
                                 const std::unordered_set<int>& features,
                                 std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) = 0;

    virtual void prune(Dataset* train_ds, Dataset* val_ds,
                       std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained) = 0;

    int predict(const std::vector<int>& sample);

public:
    void train(Dataset* train_ds, Dataset* val_ds);
    void test(Dataset* test_ds);
};

#endif
