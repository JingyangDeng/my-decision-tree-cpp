#ifndef _CSV_READER_H
#define _CSV_READER_H

#include <string>
#include <vector>

class CSVReader {
private:
    std::string path;
    std::vector<std::vector<std::string>> data;
    void load();

public:
    CSVReader(std::string path);
    const std::string& get_path();
    const std::vector<std::vector<std::string>>& get_data();
};

#endif
