#ifndef _MODEL_H
#define _MODEL_H

#include <vector>

#include "../utils/dataset.h"

class Model {
public:
    virtual void train(Dataset* train_ds, Dataset* val_ds) = 0;
    virtual void test(Dataset* test_ds) = 0;
};

#endif
