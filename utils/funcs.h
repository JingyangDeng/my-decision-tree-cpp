#ifndef _FUNCS_H
#define _FUNCS_H

#include <unordered_map>
#include <unordered_set>

#include "dataset.h"
#include "math.h"

int find(std::unordered_map<int, double>& map, double target);

void assign(Dataset* train_ds, const std::unordered_set<int>& indices, int f,
            std::unordered_map<int, std::unordered_set<int>>& new_indices);

bool label_is_same(Dataset* train_ds, const std::unordered_set<int>& indices);

int most_label(Dataset* train_ds, const std::unordered_set<int>& indices);

#endif