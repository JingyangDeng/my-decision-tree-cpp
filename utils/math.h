#ifndef _ENTROPY_H
#define _ENTROPY_H

#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../utils/dataset.h"

double nlogn(double p);

double entropy(std::unordered_map<int, int>& pmap, int sum);

double cond_entropy(std::unordered_map<int, int>& fmap, std::unordered_map<int, std::unordered_map<int, int>>& cnt,
                    int sum);

double info_gain_ratio(Dataset* train_ds, const std::unordered_set<int>& indices, int feature);

double loss_entropy(Dataset* train_ds, const std::unordered_set<int>& indices);

double gini(std::unordered_map<int, int>& pmap, int sum);

double cond_gini(std::unordered_map<int, int>& fmap, std::unordered_map<int, std::unordered_map<int, int>>& cnt,
                 int sum);

double split_cond_gini(Dataset* train_ds, const std::unordered_set<int>& indices, int feature, int value);

double loss_gini(Dataset* train_ds, const std::unordered_set<int>& indices);

int select_split(Dataset* train_ds, const std::unordered_set<int>& indices, int f, std::unordered_set<int>& selected_values);

void count(Dataset* train_ds, const std::unordered_set<int>& indices, int f, std::unordered_map<int, std::unordered_map<int, int>>& cnt, std::unordered_map<int, int>& fmap);

void count(Dataset* train_ds, const std::unordered_set<int>& indices, int f, std::unordered_map<int, std::unordered_map<int, int>>& cnt, std::unordered_map<int, int>& fmap, std::unordered_map<int, int>& lmap);

#endif
