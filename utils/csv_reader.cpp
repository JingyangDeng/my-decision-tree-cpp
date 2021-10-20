#include "csv_reader.h"

#include <fstream>
#include <sstream>

CSVReader::CSVReader(std::string path) {
    this->path = path;
    data.clear();
    this->load();
}

void CSVReader::load() {
    std::ifstream infile(path, std::ios::in);
    std::string line_str = {};
    while (getline(infile, line_str)) {
        std::stringstream ss(line_str);
        std::string str;
        std::vector<std::string> line = {};
        while (getline(ss, str, ','))
            line.emplace_back(str);
        data.emplace_back(line);
    }
}

const std::string& CSVReader::get_path() {
    return path;
}

const std::vector<std::vector<std::string>>& CSVReader::get_data() {
    return data;
}
