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

double loss(Dataset* train_ds, const std::unordered_set<int>& indices);

#endif
