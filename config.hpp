//
//  config.hpp
//  discrete_simulator
//
//  Created by Fatemeh Fardno on 2022-07-01.
//

#ifndef config_h
#define config_h
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <sstream>
#include <istream>
#include <iterator>
 
using namespace std;

class Config {
public:
    Config(){}
    bool contains(string key) const;
    void set(string key, string value);
    string get(string key, string default_val="") const;
    int getInteger(string key, int default_val=0) const;
    float getFloat(string key, float default_val=0.0f) const;
    void loadConfigFile(string file_name);
    vector<string> getVector(string key) const;
    vector<double> getDoubleVector(string key, vector<double> default_vector) const;
    friend ostream& operator<<(ostream& os, const Config& c);

private:
    unordered_map<string, string> settings;
};
#endif //DLB_IMPLEMENTATION_CONFIG_H

