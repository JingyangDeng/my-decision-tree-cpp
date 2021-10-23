#ifndef _CART_H
#define _CART_H

#include "decision_tree.h"
#define EPS_GINI 5e-2
#define MIN_LEAF 1

class DecisionTreeCART : public DecisionTree {
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
