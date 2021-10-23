#ifndef _FUNCS_H
#define _FUNCS_H

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "dataset.h"
#include "math.h"

int find(std::unordered_map<int, double>& map, double target);

std::pair<int, int> find(std::unordered_map<int, std::unordered_map<int, double>>& gidx, double gini_min);

void assign(Dataset* train_ds, const std::unordered_set<int>& indices, int f,
            std::unordered_map<int, std::unordered_set<int>>& new_indices);

void assign(Dataset* train_ds, const std::unordered_set<int>& indices, std::pair<int, int> p,
            std::unordered_map<int, std::unordered_set<int>>& new_indices);

void assign(Dataset* train_ds, const std::unordered_set<int>& indices, int f, const std::unordered_set<int>& f_values,
            std::unordered_map<int, std::unordered_set<int>>& new_indices);

bool label_is_same(Dataset* train_ds, const std::unordered_set<int>& indices);

int most_label(Dataset* train_ds, const std::unordered_set<int>& indices);

#endif
