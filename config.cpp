//
//  config.cpp
//  discrete_simulator
//
//  Created by Fatemeh Fardno on 2022-07-01.
//

#include "config.hpp"

using namespace std;

string Config::get(string key, string default_val) const {
    if(contains(key) == true)
        return settings.find(key)->second;
    return default_val;
}

int Config::getInteger(string key, int default_val) const {
    if(contains(key) == true){
        return atoi(settings.find(key)->second.c_str());
    }
    return default_val;
}
float Config::getFloat(string key, float default_val) const {
    if(contains(key) == true){
        return atof(settings.find(key)->second.c_str());
    }
    return default_val;
}
void Config::set(string key, string value) {
    settings[key] = value;
}

void Config::loadConfigFile(string file_name) {
    string directory = "/Users/fatemehfardno/Desktop/code/discrete_simulator/discrete_simulator/Configfiles/";
    
    ifstream config_file(directory+file_name, ios::binary);
    if(config_file.is_open() == false){
        cerr<< "failed to open config file\n";
    } else {
        string key_value;
        while (getline(config_file, key_value)){
            auto split_pos = key_value.find("=");
            if(split_pos != string::npos){
                settings[key_value.substr(0, split_pos)] = key_value.substr(split_pos+1);
            }
        }
    }
}


bool Config::contains(string key) const {
    if(settings.find(key) != settings.end())
        return true;
    return false;
}

ostream& operator<<(ostream& os, const Config& c){
    for(auto& key_value:c.settings)
        os << key_value.first << "=" << key_value.second << "\n";
    return os;
}


vector<string> Config::getVector(string key) const {
    auto str_val = get(key, "");
    stringstream ss(str_val);
    istream_iterator<string> begin(ss);
    istream_iterator<string> end;
    vector<string> res(begin, end);
    return res;
}

vector<double> Config::getDoubleVector(string key, vector<double> default_vector) const {
    vector<double> res;
    auto strRes = getVector(key);
    //assert (strRes.size() == default_vector.size());
    if (strRes.size() == 0)
        return default_vector;
    res.reserve(strRes.size());
    transform(strRes.begin(), strRes.end(), back_inserter(res),
            [](string const& val) {return stod(val);});
    return res;
}
