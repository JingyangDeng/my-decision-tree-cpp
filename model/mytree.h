#ifndef _MYTREE_H
#define _MYTREE_H

#include "decision_tree.h"
#define ALPHA_MYTREE 1.
#define EPS_INR_MYTREE 5e-2

class MyDecisionTree : public DecisionTree {
protected:
    void dfs(TreeNode* root, Dataset* train_ds, Dataset* val_ds,
             std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained);

    TreeNode* build_tree(Dataset* train_ds, const std::unordered_set<int>& indices,
                         const std::unordered_set<int>& features,
                         std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained);

    void prune(Dataset* train_ds, Dataset* val_ds,
               std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained);
};

#endif
