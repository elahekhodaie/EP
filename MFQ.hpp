//
//  header.hpp
//  discrete_simulator
//
//  Created by Fatemeh Fardno on 2022-07-01.
//

#ifndef header_hpp
#define header_hpp


#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <random>
#include <functional>
#include <queue>
#include <vector>
#include <cassert>
#include <numeric>
#include <cmath>
#include <math.h>
#include <chrono>
#include "config.hpp"


using namespace std;

enum ExecutionDistribution {CONSTANT, GEOM};

extern ExecutionDistribution getExecutionDist(string a);

enum InterArrivalTimeDistribution {GEOMETRIC};

extern InterArrivalTimeDistribution getInterArrivalTimeDistribution(string a);

enum LearningAlgorithm {MFQ};

extern LearningAlgorithm getLearningAlgorithm(string a);

class Distribution {
public:
    Distribution() = default;
    virtual double getNext () = 0;
    double operator()() {
        return getNext();
    }

protected:
    default_random_engine generator;

};

class Geometric: public Distribution {
public:
    explicit Geometric (double lambda) : dist(lambda) {
        generator.seed(chrono::system_clock::now().time_since_epoch().count());
    }
    double getNext () override {
        return dist(generator);
    }

private:
    geometric_distribution<> dist;

};

class Constant: public Distribution {
public:
    explicit Constant (double v) : value(v) {}
    double getNext () override {
        return value;
    }

private:
    double value;
};

class Learning {
public:
    explicit Learning(int numOfC): numOfChoices(numOfC) {}
    virtual int predict () = 0;
    virtual void updateWeights (double reward, int choice) = 0;
    virtual void setState(vector<double> state) = 0;
    virtual void updateGradientSum(double reward, int choice) = 0;
    virtual double getBatchSize() = 0;
    virtual void resetGradientSum() = 0;
    virtual vector<double> getProbabilities() = 0;
    virtual void setMeanAction(vector<double> mean_action) = 0;
    virtual void updateF(int time, vector<double> mean_action, int numPackets) = 0;
    virtual void getDisabledServer(vector<int> disabled) = 0;
    virtual vector<double> getBasis(int choice, vector<double> state, vector<double> mean_action) = 0;
    virtual int getID() = 0;
    virtual vector<double> getState() = 0;
    virtual vector<double> getPreviousState() = 0;
    virtual double getReward() = 0;
    virtual void setReward(double reward_val) = 0;
    virtual bool getF() = 0;
    virtual bool getAdam() = 0;


protected:
    int numOfChoices;
    vector<double> probability;
};

int constexpr IntPower(const int x, const int p)
{
  if (p == 0) return 1;
  if (p == 1) return x;

  int tmp = IntPower(x, p/2);
  if ((p % 2) == 0) { 
    return tmp * tmp; 
  }
  else { return x * tmp * tmp; }
}



class MFQ: public Learning{
public:
    MFQ (int number_clients, int ID, int numOfC, double beta, double batch_size, double discountFactor, double learningRate, int capacitty, double alphaa, double momentParam, double rmsParam) :numOfClients(number_clients), id(ID), Learning(numOfC), Beta(beta), batchSize(batch_size), discount_factor(discountFactor), learning_rate(learningRate), capacity(capacitty), alpha_theta(alphaa), momentumParameter(momentParam), rmsParameter(rmsParam){
        
        generator.seed(chrono::system_clock::now().time_since_epoch().count());
        normalDist = new normal_distribution<double>(0.0, 1.0/sqrt(numOfChoices));
        distribution = new uniform_real_distribution<double>(0.0, 1.0);
        
        I = 1.0;
        
        bias = 1.0;
        
        disabled_server.resize(numOfChoices, 0);
        probability.resize(numOfChoices, 1.0/numOfChoices);
        previous_state.resize(numOfChoices, 0.0f);
        mean_action.resize(numOfChoices, 0.0f);        
        state.resize(numOfChoices, 0.0);
        
        for(int i=0; i<numOfChoices; i++){
            disabled_server[i] = 0;
            probability[i] = 1.0/numOfChoices;
            previous_state[i] = 0.0;
            mean_action[i] = 0.0;
            state[i] = 0.0;
        }
        
        assert(disabled_server.size()==numOfChoices);
        assert(probability.size()==numOfChoices);
        assert(previous_state.size()==numOfChoices);
        assert(mean_action.size()==numOfChoices);
        assert(state.size()==numOfChoices);
    
        
        f = new double*[IntPower(capacity,numOfChoices)];
        
        for(int i = 0; i<IntPower(capacity,numOfChoices); i++){
            f[i] = new double[numOfChoices];
        }
        
        for(int i = 0; i<IntPower(capacity,numOfChoices); i++){
            for(int j=0; j<numOfChoices; j++){
                f[i][j] = 1.0/numOfChoices;
            }
        }
        
        
        if (fAdded==false) theta_size = IntPower(numOfChoices,2) + 1;
        else if (fAdded==true) theta_size = IntPower(numOfChoices,3) + IntPower(numOfChoices,2) + 1;

        theta.resize(theta_size, 0.0);
        V_adam.resize(theta_size, 0.0);
        S_adam.resize(theta_size, 0.0);
        GradientSum.resize(theta_size, 0.0);
        vAdam.resize(theta_size, 0.0);
        mAdam.resize(theta_size, 0.0);
        
        
        for (int j = 0; j<theta_size; j++){
            theta[j] = 0.0;
            V_adam[j] = 0.0;
            S_adam[j] = 0.0;
            vAdam[j] = 0.0;
            mAdam[j] = 0.0;
            GradientSum[j] = 0.0;
        }
        
    }
    
    int predict() override;
    
    void setState(vector<double> state) override;
    
    void setMeanAction(vector<double> mean_action) override;
    
    void updateF(int time, vector<double> mean_action, int numPackets) override;
    
    void updateGradientSum(double reward, int choice) override;
    
    double getBatchSize() override;
    
    void resetGradientSum() override;
    
    void updateWeights (double reward, int choice) override;

    void getDisabledServer(vector<int> disabled) override;
        
    vector<double> getProbabilities() override;
    
    vector<double> getBasis(int choice, vector<double> state, vector<double> mean_action) override;
    
    int getID() override;
    
    vector<double> getState() override;
    
    vector<double> getPreviousState() override;
    
    double getReward() override;
    
    void setReward(double reward_val) override;
    
    bool getF() override;
    
    bool getAdam() override;
    
private:
    bool Adam = true;
    bool fAdded = false;
    int numOfClients;
    int id;
    double Beta;
    double bias;
    double batchSize;
    vector<double> theta;
    vector<double> V_adam;
    vector<double> S_adam;
    vector<double> mAdam;
    vector<double> vAdam;
    int theta_size;
    double** f;
    double I;
    double alpha_theta;
    double discount_factor;
    double learning_rate;
    double reward = 0.0;
    double previous_reward = 0.0;
    int choice = 0;
    int previous_choice = 0;
    vector<double> GradientSum;
    vector<double> state;
    vector<double> previous_state;
    vector<double> mean_action;
    vector<int> disabled_server;
    int capacity;
    double momentumParameter = 0.9;
    double rmsParameter = 0.999;
    default_random_engine generator;
    uniform_real_distribution<> * distribution;
    normal_distribution<> * normalDist;
    int counter = 0;
};


class Task;
class Server;

class Client {
public:
    Client(int clientID, Learning * learnerPtr, Distribution * extDistPtr, double averageLatencyrate)
    : id(clientID), learner(learnerPtr), execDistribution(extDistPtr), avgLatencyRate(averageLatencyrate){
        previousArrival = 0.0;
        avgLatency = 0;
        numArrivals = 0;
        latencyVector.resize(1,0.0);
    }

    int getId () { return id; }
    
    int getNumArrivals() { return numArrivals; }
    
    void updateNumArrivals() {
        numArrivals += 1;
    }
    
    void setLatency(double latencyValue){
        latency = latencyValue;
    }
    
    double getlatency() { return latency; }

    Learning * getLearner () { return learner; }
        
    double getAverageLatency() { return avgLatency; }
    
    void updateAvgLatency(double latency){
        avgLatency = avgLatency * avgLatencyRate + (1-avgLatencyRate) * latency;
    }
    
    void updateLatencyVector( double newLatency){
        latencyVector.push_back(newLatency);
    }
    
    vector<double> getLatencyVector(){ return latencyVector;}
    
    double getTaskExecTime(){
        return execDistribution->getNext();
    }
    
    double getCounter(){
        return counter;
    }
    
    void updateCounter(){
        counter += 1;
    }
    
    void resetPacketReceived(){
        packetReceived = false;
    }
    
    void updatePacketReceived(){
        packetReceived = true;
    }
    
    bool getPacketReceived(){
        return packetReceived;
    }
    
    void updateBatchCounter(){
        batchCounter += 1;
    }
    
    void resetBatchCounter(){
        batchCounter = 0;
    }
    
    int getBatchCounter(){
        return batchCounter;
    }
    
    
    
private:
    int id;
    double previousArrival;
    Learning * learner;
    //Distribution * intADistribution;
    Distribution * execDistribution;
    double avgLatencyRate;
    double avgLatency;
    vector<double> latencyVector;
    double latency;
    int numArrivals;
    double counter = 1.0;
    bool packetReceived = false;
    int batchCounter = 0;


};

class Task{
public:
    Task(int arrival, double exec, Client * clientPtr)
    : arrivalTime(arrival), executionTime(exec), client(clientPtr) {
        originalExecutionTime = executionTime;
    }

    struct compareTasks {
        bool operator()(Task* const& left, Task* const& right) {
            if (left->arrivalTime != right->arrivalTime)
                return left->arrivalTime > right->arrivalTime;
            else
                return left->client->getId() > right->client->getId();
        }
    };

    double getArrivalTime () { return arrivalTime; }
    double getExecutionTime() { return executionTime; }
    double getOriginalExecutionTime() { return originalExecutionTime;}
    void setExecTime(double execTime) { executionTime = execTime;}
    Client * getClient () { return client; }
    void setServer(Server * serverPtr){ server = serverPtr; }
    Server * getServer () { return server; }
    

private:
    int arrivalTime;
    double executionTime;
    double originalExecutionTime;
    

    Client * client;
    Server * server;
};


class Server{
public:
    Server(int serverID, double serviceRate, double averageServiceTimeRate, double averageWaitTimeRate ) : id(serverID), rate(serviceRate), avgServiceTimeRate(averageServiceTimeRate), avgWaitTimeRate(averageWaitTimeRate){
        avgServiceTime = 0.0;
        avgWaitTime = 0.0;
        numDepartures = 0;
    }
    
    int getId () { return id; }

    double getExpectedWaitTime () {
        return taskQueue.size() * avgServiceTime;
    }
    
    double getAvgServiceTime(){
        return avgServiceTime;
    }
    
    int getQueueSize(){
        return int(taskQueue.size());
    }

    void emplaceTask(Task * task) {
        taskQueue.emplace(task);
        busy = true;
    }
    
    double getAvgWaitTime(){
        return avgWaitTime;
    }
    
    double getRate(){
        return rate;
    }
    
    int getNumDepartures(){
        return numDepartures;
    }
    
    Task * getTop(){
        return taskQueue.top();
    }
    
    // This function is called on departure of a task on top of the queue
    Task * serveNextTask (double currentTime);

    bool isBusy () { return busy; }

private:
    int id;
    double rate;
    bool busy = false;
    int numDepartures=0;
    double avgServiceTime;
    double avgServiceTimeRate;
    double avgWaitTime;
    double avgWaitTimeRate;
    priority_queue<Task *, vector<Task *>, Task::compareTasks> taskQueue;
};

class Simulator{
public:

    Simulator (Config * config);
    ~Simulator() {
        for (auto p : listOfClients)
            delete p;
        listOfClients.clear();
        for (auto p : listOfServers)
            delete p;
        listOfServers.clear();
        
    }
    
    void discreteRun(int configIndex);
    
    void testRun1(int configIndex);
    
    void testRun2(int configIndex);
    
    void testRun3(int configIndex);
    
    void testRun4(int configIndex);
    
    void testRun5(int configIndex);

    void printResults();
    
    void saveResults(int configIndex);
    
    void createFiles();
    
    double percentile(vector<double> vectorIn)
    {
        auto nth = vectorIn.begin() + (percent*vectorIn.size())/100;
        nth_element(vectorIn.begin(), nth, vectorIn.end());
        return *nth;
    }

        
private:
    int simulationTime;
    int numClients;
    int numServers;
    int numRecievedPackets = 0;
    double current_time = 0.0;
    int current_dtime = 0;
    int counter_psga = 0;
    vector<Client*> listOfClients;
    vector<Server*> listOfServers;
    double samplingInterval;
    int printLevel;
    double rewardEpsilon;
    double percent;
    double normalizationFactor;
    string learning_algorithm;
    int capacity;
    vector<double> state;
    vector<int> disabled_servers;
    default_random_engine generator;
    uniform_real_distribution<> * distribution;
    vector<double> arrival_prob;
    vector<double> mean_action;
    vector<double> rewardBuffer;
    vector<int> choiceBuffer;
};


#endif /* header_hpp */
