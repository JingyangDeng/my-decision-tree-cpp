#ifndef _C4_5_H
#define _C4_5_H

#include "decision_tree.h"
#define EPS 0.05
#define ALPHA 1.

class DecisionTreeC4_5 : public DecisionTree {
protected:
    void dfs(TreeNode* root, Dataset* train_ds, std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained);

    TreeNode* build_tree(Dataset* train_ds, const std::unordered_set<int>& indices,
                         const std::unordered_set<int>& features,
                         std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained);

    void prune(Dataset* train_ds, std::unordered_map<TreeNode*, std::unordered_set<int>>& data_contained);
};

#endif
