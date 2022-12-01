//
//  main.cpp
//  discrete_simulator
//
//  Created by Fatemeh Fardno on 2022-07-01.
//

#include <iostream>
#include <string>
#include <queue>
#include <random>
#include <functional>
#include <algorithm>
#include <queue>
#include <vector>
#include <cassert>
#include <numeric>
#include <cmath>
#include <Python/Python.h>
#include "config.hpp"
#include "MFQ.hpp"


using namespace std;

int main() {

    int configCount = 1;

    for(int i=0; i<configCount; i++){
        Config * config = new Config();
        char configIndex = (i+1) + '0';
        string configFile = "ConfigFile";
        configFile += configIndex;
        string filename = configFile + ".txt";

        config->loadConfigFile(filename);
        cout<<filename<<endl;
        Simulator simulator(config);
        
//        simulator.testRun1(i+1); // There are two states, when players play a1 in s1 get reward 80, otherwise get 100 and got to s2, where all rewards are 1. each agent has its own state
        
//        simulator.testRun2(i+1); // There are two states, when all players play a1 in s1, they get 100, otherwise they get 1. The same happens for s2. State transitions are independent from joint actions. State is global
        
//        simulator.testRun3(i+1);  // There is only one global state. When all players take a1 in s1 or a2 they get 100. otherwise they get 1
        
//        simulator.testRun4(i+1);  // There are two states. At s1 actions should be the same to get 100, in s2 they should be different to get 1. State transitions are independent from joint actions. State is global
                
        simulator.discreteRun(i+1);

    }

    
    return 0;
}

