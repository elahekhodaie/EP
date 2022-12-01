//
//  PSGA.cpp
//  PSGA
//
//  Created by Fatemeh Fardno on 2022-07-17.
//

#include "PSGA.hpp"


using namespace std;

ExecutionDistribution getExecutionDist(string a) {
    if (a == "CONSTANT")
        return CONSTANT;
    if ( a == "GEOM")
        return GEOM;
    exit(-1);
}

//InterArrivalTimeDistribution getInterArrivalTimeDistribution(string a){
//    if ( a == "GEOMETRIC")
//        return GEOMETRIC;
//    exit(-1);
//}

LearningAlgorithm getLearningAlgorithm(string a){
    if( a == "PSGA")
        return PSGA;
    exit(-1);
}


int PSGA::predict(){
    double sum = 0.0;
    int index = 0;

    for(int i=0; i<numOfChoices; i++){
        index += state[i]*pow(2,i);
    }
    
    for(int i=0; i<numOfChoices; i++){
        probability[i] = (1.0 - exploration_rate) * theta[i][index] + exploration_rate/numOfChoices;
        sum += probability[i];
    }

    assert(sum-1.0<0.000001);
    

    auto val = (*distribution)(generator);
    int choice = 0;
    double var = 0.0;
    
    for (int i = 0; i < numOfChoices; i++){
        var = var + probability[i];
        if (val <= var){
            choice = i;
            break;
 }
}
    assert(choice < numOfChoices);
    return choice;
    
}

void PSGA::setState(vector<double> stateValue){
    previous_state = state;
    state = stateValue;
    
}

void PSGA::updateGradientSum(double reward, int choice){
    
    R += reward;
    
    int index = 0;
    for(int i=0; i<numOfChoices; i++){
        index += previous_state[i]*pow(2,i);
    }
    logSum[choice][index] = logSum[choice][index] + 1.0/probability[choice];
    
}

double PSGA::getBatchSize(){
    return batchSize;
}

double PSGA::getReward(){
    return reward;
}

void PSGA::setReward(double reward_val){
    reward = reward_val;
}

void PSGA::updateWeights(double reward, int choice){
    
    updateGradientSum(reward, choice);
    
    for(int i=0; i<pow(2,numOfChoices); i++){
        double column_sum = 0.0;
        for(int j=0; j<numOfChoices; j++){
            theta[j][i] = theta[j][i] + learning_rate * logSum[j][i] * R;
            column_sum += theta[j][i];
        }
        
        for(int j=0; j<numOfChoices; j++){
            theta[j][i] = theta[j][i] /column_sum;
        }
        
    }
    
    for(int i=0; i<pow(2,numOfChoices); i++){
        double column_sum = 0.0;
        for(int j=0; j<numOfChoices; j++){
            column_sum = column_sum + theta[j][i];
        }
        
        assert(column_sum-1.0<0.0001);
    }
    
}

void PSGA::resetGradientSum(){
    R = 0.0;
    for(int i=0; i<numOfChoices; i++){
        for(int j=0; j<pow(2,numOfChoices); j++){
            logSum[i][j] = 0.0;
        }
    }
}


vector<double> PSGA::getProbabilities(){
    return probability;
}

vector<double> PSGA::getState(){
    return state;
}

int PSGA::getID(){
    return id;
}

vector<double> PSGA::getPreviousState(){
    return previous_state;
}


Task * Server::serveNextTask (double currentTime) {
    Task * task = taskQueue.top();
    assert (taskQueue.empty() == false);
    assert (task->getArrivalTime() < currentTime);
    numDepartures += 1;
    // Exponentially weighted moving average with bias correction
    // In practice we don't use bias correction because after about 10 steps the moving average is warmed up and there is no need for bias correction
    avgServiceTime = avgServiceTimeRate * avgServiceTime + ( 1 - avgServiceTimeRate ) * (task->getOriginalExecutionTime()/rate);
    avgWaitTime = avgWaitTimeRate * avgWaitTime + (1 - avgWaitTimeRate) * (currentTime - task->getArrivalTime());
    taskQueue.pop();

    if (taskQueue.empty()) {
        busy = false;
        return nullptr;
    } else {
        busy = true;
        return taskQueue.top();
    }
}

Simulator::Simulator (Config * config) {

    
    numServers = config->getInteger("numServers");
    simulationTime = config->getFloat("simulationTime");
    numClients = config->getInteger("numClients");
    samplingInterval = config->getFloat("samplingInterval", 10);
    printLevel = config->getInteger("printLevel",2);
    rewardEpsilon = config->getFloat("rewardEpsilon",0.05);
    percent = config->getFloat("percent",99);
    normalizationFactor = config->getFloat("normalizationFactor",10000);
    capacity = config->getInteger("capacity",2);
    
    double Beta = config->getFloat("Beta",1);
    double batchSize = config->getFloat("batchSize",100);
    auto momentParam = config->getFloat("momentumParameter",0.9);
    auto RMSParam = config->getFloat("RMSParameter",0.999);
    auto discount_factor = config->getFloat("discountFactor",0.99);
    
    double learningR = config->getFloat("learningRate",1.0);
    double expR = config->getFloat("explorationRate",1.0);

    
    rewardBuffer.resize(numClients, 0.0);
    choiceBuffer.resize(numClients, 0.0);
    vector<double> default_vector;
//    default_vector.resize(numClients, 0.0);
    default_vector.resize(1, 0.0);
    state.resize(numServers, 0.0);
    disabled_servers.resize(numServers, 0);

    auto executionDistVector = config->getVector("execDists");
    auto executionAvgVector = config->getDoubleVector("executionAverages", default_vector);

    auto clientsArrivalRate = config->getDoubleVector("clientsArrivalRates", default_vector);
    auto learningAlgorithmVector = config->getVector("learningAlgorithms");
    auto learningParameters = config->getDoubleVector("learningParameters", default_vector);
    
    auto alpha_theta = config->getDoubleVector("alpha_theta", default_vector);
    auto avgLatencyRate = config->getFloat("averageLatencyRate", 0.99);
    
    state.resize(numServers, 0.0);
    state[0] = 1.0;
//    state[1] = 1.0;


    
    mean_action.resize(numServers, 0.0);
        
    for (int i = 0; i < numClients; i++) {
        
        Distribution * execDist;
        switch (getExecutionDist(executionDistVector[0])) {
            case CONSTANT:
                execDist = new Constant(executionAvgVector[0]);
                break;
            case GEOM:
                execDist = new Geometric(executionAvgVector[0]);
            default:
                cout << "did not recognize the execution distribution" << endl;
                exit(-1);
        }
//        (int number_clients, int ID, int numOfC, double batch_size, double learningRate, double expRate)
        
        double arrival_rate = clientsArrivalRate[0]/numClients;
        arrival_prob.push_back(arrival_rate);
        arrival_prob.push_back(1.0 - arrival_rate);

        cout<<expR<<endl;
        Learning * lr;
        generator.seed(chrono::system_clock::now().time_since_epoch().count());
        distribution = new uniform_real_distribution<double>(0.0, 1.0);
        switch (getLearningAlgorithm(learningAlgorithmVector[0])){
            case PSGA:
                learning_algorithm = "PSGA";
                lr = new class PSGA(numClients, i, numServers, batchSize, learningR, expR);
                break;
                
            default:
                cout << "did not recognize the learning algorithm" << endl;
                exit(-1);
        }
        
        auto *client = new Client(i, lr, execDist, avgLatencyRate);
        listOfClients.push_back(client);
    }

    default_vector.resize(numServers, 0.0);
    auto serversServiceRate = config->getDoubleVector("serversServiceRate", default_vector);
    assert(serversServiceRate.size() == numServers);
    
    auto avgServiceTimeRate = config->getFloat("averageServiceTimeRate", 0.99);
    auto avgWaitTimeRate = config->getFloat("averageWaitTimeRate", 0.99);
    
    for (int i = 0; i < numServers; i++){
        auto *server = new Server(i, serversServiceRate[i], avgServiceTimeRate, avgWaitTimeRate);
        listOfServers.push_back(server);
    }
    
}


void Simulator::testRun2(int configIndex){
    
    int currentInterval = 0;
    
    for(current_time=0; current_time<simulationTime; current_time++){
        cout << "\n\n\n" << endl;
        double mean_prob_s1_a1 = 0.0;
        double mean_prob_s1_a2 = 0.0;
        
        double mean_prob_s2_a1 = 0.0;
        double mean_prob_s2_a2 = 0.0;
        
        numRecievedPackets = 0;
        
        for(int i=0; i<numClients; i++){
            rewardBuffer[i] = 0.0;
            choiceBuffer[i] = 0;
        }
        
        for(int i=0; i<numServers; i++) mean_action[i] = 0.0;

        
        for(int i =0; i<numClients; i++){
            
            Client * client = listOfClients[i];
            client->resetPacketReceived();

            auto val = (*distribution)(generator);
            double var = 0.0;
            var = var + arrival_prob[0];

            if (val <= var) client->updatePacketReceived();
            else client->resetPacketReceived();
        
            
            if(client->getPacketReceived()){
                client->updateBatchCounter();
                numRecievedPackets += 1;
                auto *task = new Task(current_time, client->getTaskExecTime(), client);
                // declares which server is going to serve the task
                int serverID = client->getLearner()->predict();
                // check if serverID is valid
                assert(serverID < numServers);
                client->updateNumArrivals();
                // pointer to the server
                auto server = listOfServers[serverID];
                // set server to the task
                task->setServer(server);
                // Add task to server taskQueue
                server->emplaceTask(task);
                // client gets expected latency as cost
                choiceBuffer[i] = serverID;
            }
        }

        int tempp = 0;
        if(state[0]==1.0 && state[1]==0.0){
            for(int i=0; i<numClients; i++){
                if(listOfClients[i]->getPacketReceived()){
                    if(choiceBuffer[i]==0) tempp += 1;
                    mean_prob_s1_a1 += listOfClients[i]->getLearner()->getProbabilities()[0];
                    mean_prob_s1_a2 += listOfClients[i]->getLearner()->getProbabilities()[1];
                }
            }
            
            if(numRecievedPackets != 0){
                mean_prob_s1_a1 = mean_prob_s1_a1/numRecievedPackets;
                mean_prob_s1_a2 = mean_prob_s1_a2/numRecievedPackets;
                
            }
            
            if(tempp == numRecievedPackets){
                cout<<"**"<<endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 100.0;
                        listOfClients[i]->getLearner()->setReward(100.0);
                }

          }
        }
            
            if(tempp != numRecievedPackets){
                cout << "!!" << endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 1.0;
                        listOfClients[i]->getLearner()->setReward(1.0);

                    }
               }

            }
            auto val = (*distribution)(generator);
            if (val <= 0.9){
                state[0] = 1.0;
                state[1] = 0.0;
            }
            else{
                state[0] = 0.0;
                state[1] = 1.0;
            }
    }
        
        else if(state[0]==0.0 && state[1]==1.0){
            for(int i=0; i<numClients; i++){
                if(listOfClients[i]->getPacketReceived()){
                    if(choiceBuffer[i]==1) tempp += 1;
                    mean_prob_s2_a1 += listOfClients[i]->getLearner()->getProbabilities()[0];
                    mean_prob_s2_a2 += listOfClients[i]->getLearner()->getProbabilities()[1];
                }
            }
            
            if(numRecievedPackets != 0){
                mean_prob_s2_a1 = mean_prob_s2_a1/numRecievedPackets;
                mean_prob_s2_a2 = mean_prob_s2_a2/numRecievedPackets;
                
            }
            
            if(tempp == numRecievedPackets){
                cout<<"**"<<endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 100.0;
                        listOfClients[i]->getLearner()->setReward(100.0);
                    }
                }
          }
            
            if(tempp != numRecievedPackets){
                cout << "!!" << endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 1.0;
                        listOfClients[i]->getLearner()->setReward(1.0);
                }
            }

        }
            
            auto val = (*distribution)(generator);
            if (val <= 0.9){
                state[0] = 0.0;
                state[1] = 1.0;
            }
            else{
                state[0] = 1.0;
                state[1] = 0.0;
            }
    }
        
        for(int i=0; i<numClients; i++) listOfClients[i]->getLearner()->setState(state);
        
        
        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;
        cout<<"Client 1, server 1 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[0]<<", Client 1, server 2 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 2, server 1 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[0]<<", Client 2, server 2 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 3, server 1 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[1]<<endl;


        
        for(int i=0; i<numClients; i++){
            
                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
            
                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
                    listOfClients[i]->getLearner()->resetGradientSum();
                    listOfClients[i]->updateCounter();
                    listOfClients[i]->resetBatchCounter();
                }
                else if(listOfClients[i]->getPacketReceived()) listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
        }
        
        // print results
        if (current_time > currentInterval * samplingInterval) {
            currentInterval++;
            this->printResults();
            this->saveResults(configIndex);
            
        }
    }
}

void Simulator::testRun3(int configIndex){
    
    int currentInterval = 0;
    
    for(current_time=0; current_time<simulationTime; current_time++){
        cout << "\n\n\n" << endl;
        double mean_prob_s1_a1 = 0.0;
        double mean_prob_s1_a2 = 0.0;
        
        numRecievedPackets = 0;
        
        for(int i=0; i<numClients; i++){
            rewardBuffer[i] = 0.0;
            choiceBuffer[i] = 0;
        }
        
        for(int i =0; i<numClients; i++){
            
            Client * client = listOfClients[i];
            client->resetPacketReceived();

            auto val = (*distribution)(generator);
            double var = 0.0;
            var = var + arrival_prob[0];
//            if (val <= var) packetReceived = true;
//            else packetReceived = false;
            if (val <= var) client->updatePacketReceived();
            else client->resetPacketReceived();
        
            
            if(client->getPacketReceived()){
                client->updateBatchCounter();
                numRecievedPackets += 1;
                auto *task = new Task(current_time, client->getTaskExecTime(), client);
                // declares which server is going to serve the task
                int serverID = client->getLearner()->predict();
                // check if serverID is valid
                assert(serverID < numServers);
                client->updateNumArrivals();
                // pointer to the server
                auto server = listOfServers[serverID];
                // set server to the task
                task->setServer(server);
                // Add task to server taskQueue
                server->emplaceTask(task);
                // client gets expected latency as cost
                choiceBuffer[i] = serverID;
            }
            
        }

        int tempp1 = 0;
        int tempp2 = 0;
        
        if(state[0]==1.0 && state[1]==1.0){
            for(int i=0; i<numClients; i++){
                if(listOfClients[i]->getPacketReceived()){
                    if(choiceBuffer[i]==0) tempp1 += 1;
                    if(choiceBuffer[i]==1) tempp2 += 1;
                    mean_prob_s1_a1 += listOfClients[i]->getLearner()->getProbabilities()[0];
                    mean_prob_s1_a2 += listOfClients[i]->getLearner()->getProbabilities()[1];
                }
            }
            
            if(numRecievedPackets != 0){
                mean_prob_s1_a1 = mean_prob_s1_a1/numRecievedPackets;
                mean_prob_s1_a2 = mean_prob_s1_a2/numRecievedPackets;
                
            }
            
            if(tempp1 == numRecievedPackets){
                cout<<"**"<<endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 100.0;
                        listOfClients[i]->getLearner()->setReward(100.0);
                    }
                }
          }
            else if(tempp2 == numRecievedPackets){
                cout<<"$$"<<endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 100.0;
                        listOfClients[i]->getLearner()->setReward(100.0);
                    }
                }
          }
            
            else if(tempp1 != numRecievedPackets && tempp2 != numRecievedPackets){
                cout << "!!" << endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 1.0;
                        listOfClients[i]->getLearner()->setReward(1.0);
                    }
               }

            }
            
            state[0] = 1.0;
            state[1] = 1.0;

    }
        
        
        for(int i=0; i<numClients; i++) listOfClients[i]->getLearner()->setState(state);
                
        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;

        cout<<"Client 1, server 1 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[0]<<", Client 1, server 2 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 2, server 1 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[0]<<", Client 2, server 2 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 3, server 1 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 4, server 1 prob: "<<listOfClients[3]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[3]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 5, server 1 prob: "<<listOfClients[4]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[4]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 6, server 1 prob: "<<listOfClients[5]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[5]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 7, server 1 prob: "<<listOfClients[6]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[6]->getLearner()->getProbabilities()[1]<<endl;


        
        for(int i=0; i<numClients; i++){
            
                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
            
                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
                    listOfClients[i]->getLearner()->resetGradientSum();
                    listOfClients[i]->updateCounter();
                    listOfClients[i]->resetBatchCounter();
                }
                else if(listOfClients[i]->getPacketReceived()) listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
        }
        
        // print results
        if (current_time > currentInterval * samplingInterval) {
            currentInterval++;
            this->printResults();
            this->saveResults(configIndex);
            
        }
    }
}

void Simulator::testRun4(int configIndex){
    
    int currentInterval = 0;
    
    for(current_time=0; current_time<simulationTime; current_time++){
        cout << "\n\n\n" << endl;
        double mean_prob_s1_a1 = 0.0;
        double mean_prob_s1_a2 = 0.0;
        
        double mean_prob_s2_a1 = 0.0;
        double mean_prob_s2_a2 = 0.0;
        
        numRecievedPackets = 0;
        
        for(int i=0; i<numClients; i++){
            rewardBuffer[i] = 0.0;
            choiceBuffer[i] = 0;
        }
        
        
        for(int i =0; i<numClients; i++){
            
            Client * client = listOfClients[i];
            client->resetPacketReceived();

            auto val = (*distribution)(generator);
            double var = 0.0;
            var = var + arrival_prob[0];
//            if (val <= var) packetReceived = true;
//            else packetReceived = false;
            if (val <= var) client->updatePacketReceived();
            else client->resetPacketReceived();
        
            
            if(client->getPacketReceived()){
                client->updateBatchCounter();
                numRecievedPackets += 1;
                auto *task = new Task(current_time, client->getTaskExecTime(), client);
                // declares which server is going to serve the task
                int serverID = client->getLearner()->predict();
                // check if serverID is valid
                assert(serverID < numServers);
                client->updateNumArrivals();
                // pointer to the server
                auto server = listOfServers[serverID];
                // set server to the task
                task->setServer(server);
                // Add task to server taskQueue
                server->emplaceTask(task);
                // client gets expected latency as cost
                choiceBuffer[i] = serverID;
                mean_action[serverID] += 1.0;
            
            }
            
        }
        assert(numRecievedPackets==numClients);

        int tempp1 = 0;
        int tempp2 = 0;
        if(state[0]==1.0 && state[1]==0.0){
            for(int i=0; i<numClients; i++){
                if(listOfClients[i]->getPacketReceived()){
                    if(choiceBuffer[i]==0) tempp1 += 1;
                    if(choiceBuffer[i]==1) tempp2 += 1;

                    mean_prob_s1_a1 += listOfClients[i]->getLearner()->getProbabilities()[0];
                    mean_prob_s1_a2 += listOfClients[i]->getLearner()->getProbabilities()[1];
                }
            }
            
            assert(tempp1 + tempp2==numRecievedPackets);
            
            if(numRecievedPackets != 0){
                mean_prob_s1_a1 = mean_prob_s1_a1/numRecievedPackets;
                mean_prob_s1_a2 = mean_prob_s1_a2/numRecievedPackets;
                
            }
            
            if(tempp1 == numRecievedPackets || tempp2 == numRecievedPackets){
                cout<<"**"<<endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 100.0;
                        listOfClients[i]->getLearner()->setReward(100.0);
                    }
                }
          }
            
            else {
                cout << "!!" << endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 1.0;
                        listOfClients[i]->getLearner()->setReward(1.0);
                    }

               }
            }
            auto val = (*distribution)(generator);
            if (val <= 0.9){
                state[0] = 1.0;
                state[1] = 0.0;
            }
            else{
                state[0] = 0.0;
                state[1] = 1.0;
            }
    }
        
        else if(state[0]==0.0 && state[1]==1.0){
            for(int i=0; i<numClients; i++){
                if(listOfClients[i]->getPacketReceived()){
                    if(choiceBuffer[i]==0) tempp1 += 1;
                    if(choiceBuffer[i]==1) tempp2 += 1;
                    mean_prob_s2_a1 += listOfClients[i]->getLearner()->getProbabilities()[0];
                    mean_prob_s2_a2 += listOfClients[i]->getLearner()->getProbabilities()[1];
                }
            }
            
            if(numRecievedPackets != 0){
                mean_prob_s2_a1 = mean_prob_s2_a1/numRecievedPackets;
                mean_prob_s2_a2 = mean_prob_s2_a2/numRecievedPackets;
                
            }
            
            if(tempp1 == numRecievedPackets || tempp2 == numRecievedPackets){
                cout<<"**"<<endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 1.0;
                        listOfClients[i]->getLearner()->setReward(1.0);
                    }
                    
                }
          }
            
            else {
                cout << "!!" << endl;
                for(int i=0; i<numClients; i++){
                    if(listOfClients[i]->getPacketReceived()) {
                        rewardBuffer[i] = 100.0;
                        listOfClients[i]->getLearner()->setReward(100.0);
                    }
                }

            }
            
            auto val = (*distribution)(generator);
            if (val <= 0.9){
                state[0] = 0.0;
                state[1] = 1.0;
            }
            else{
                state[0] = 1.0;
                state[1] = 0.0;
            }
    }
        
        for(int i=0; i<numClients; i++) listOfClients[i]->getLearner()->setState(state);
                
        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;
        cout<<"Client 1, server 1 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[0]<<", Client 1, server 2 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 2, server 1 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[0]<<", Client 2, server 2 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[1]<<endl;


        
        for(int i=0; i<numClients; i++){
            
                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
            
                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
                    listOfClients[i]->getLearner()->resetGradientSum();
                    listOfClients[i]->updateCounter();
                    listOfClients[i]->resetBatchCounter();
                }
                else if(listOfClients[i]->getPacketReceived()) listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
        }

        
        // print results
        if (current_time > currentInterval * samplingInterval) {
            currentInterval++;
            this->printResults();
            this->saveResults(configIndex);
            
        }
    }
}

    
//void Simulator::testRun5(int configIndex){
//
//    int currentInterval = 0;
//
//    for(current_time=0; current_time<simulationTime; current_time++){
//        cout << "\n\n\n" << endl;
//        double mean_prob_s1_a1 = 0.0;
//        double mean_prob_s1_a2 = 0.0;
//
//        double mean_prob_s2_a1 = 0.0;
//        double mean_prob_s2_a2 = 0.0;
//
//        numRecievedPackets = 0;
//
//        for(int i=0; i<numClients; i++){
//            rewardBuffer[i] = 0.0;
//            choiceBuffer[i] = 0;
//        }
//
//        for(int i=0; i<numServers; i++) mean_action[i] = 0.0;
//
//
//        for(int i =0; i<numClients; i++){
//
//            Client * client = listOfClients[i];
//            client->resetPacketReceived();
//
//            auto val = (*distribution)(generator);
//            double var = 0.0;
//            var = var + arrival_prob[0];
////            if (val <= var) packetReceived = true;
////            else packetReceived = false;
//            if (val <= var) client->updatePacketReceived();
//            else client->resetPacketReceived();
//
//
//            if(client->getPacketReceived()){
//                client->updateBatchCounter();
//                numRecievedPackets += 1;
//                auto *task = new Task(current_time, client->getTaskExecTime(), client);
//                // declares which server is going to serve the task
//                int serverID = client->getLearner()->predict();
//                // check if serverID is valid
//                assert(serverID < numServers);
//                client->updateNumArrivals();
//                // pointer to the server
//                auto server = listOfServers[serverID];
//                // set server to the task
//                task->setServer(server);
//                // Add task to server taskQueue
//                server->emplaceTask(task);
//                // client gets expected latency as cost
//                choiceBuffer[i] = serverID;
//                mean_action[serverID] += 1.0;
//
//            }
//
//        }
//
//        int tempp = 0;
//        if(state[0]==1.0 && state[1]==0.0){
//            for(int i=0; i<numClients; i++){
//                if(listOfClients[i]->getPacketReceived()){
//                    if(choiceBuffer[i]==0) tempp += 1;
//                    mean_prob_s1_a1 += listOfClients[i]->getLearner()->getProbabilities()[0];
//                    mean_prob_s1_a2 += listOfClients[i]->getLearner()->getProbabilities()[1];
//                }
//            }
//
//            if(numRecievedPackets != 0){
//                mean_prob_s1_a1 = mean_prob_s1_a1/numRecievedPackets;
//                mean_prob_s1_a2 = mean_prob_s1_a2/numRecievedPackets;
//
//            }
//
//            if(tempp == numRecievedPackets){
//                cout<<"**"<<endl;
//                for(int i=0; i<numClients; i++){
//                    if(listOfClients[i]->getPacketReceived()) rewardBuffer[i] = 80.0;
//                }
//                auto val = (*distribution)(generator);
//                if (val <= 0.9){
//                    state[0] = 1.0;
//                    state[1] = 0.0;
//                }
//                else{
//                    state[0] = 0.0;
//                    state[1] = 1.0;
//                }
//          }
//
//            if(tempp != numRecievedPackets){
//                cout << "!!" << endl;
//                for(int i=0; i<numClients; i++){
//                    if(listOfClients[i]->getPacketReceived()) rewardBuffer[i] = 100.0;
//               }
//                state[0] = 0.0;
//                state[1] = 1.0;
//            }
//    }
//
//        else if(state[0]==0.0 && state[1]==1.0){
//            for(int i=0; i<numClients; i++){
//                if(listOfClients[i]->getPacketReceived()){
//                    mean_prob_s2_a1 += listOfClients[i]->getLearner()->getProbabilities()[0];
//                    mean_prob_s2_a2 += listOfClients[i]->getLearner()->getProbabilities()[1];
//                }
//            }
//
//            if(numRecievedPackets != 0){
//                mean_prob_s2_a1 = mean_prob_s2_a1/numRecievedPackets;
//                mean_prob_s2_a2 = mean_prob_s2_a2/numRecievedPackets;
//
//            }
//
//            for(int i=0; i<numClients; i++){
//                if(listOfClients[i]->getPacketReceived()) rewardBuffer[i] = 1.0;
//            }
//
//            auto val = (*distribution)(generator);
//            if (val <= 0.9){
//                state[0] = 0.0;
//                state[1] = 1.0;
//            }
//            else{
//                state[0] = 1.0;
//                state[1] = 0.0;
//            }
//    }
//
//        for(int i=0; i<numClients; i++) listOfClients[i]->getLearner()->setState(state);
//
//        for(int i=0;i<numServers; i++){
//            if(numRecievedPackets != 0) mean_action[i] = mean_action[i]/numRecievedPackets;
//        }
//
//
//
//
//        for(int i=0; i<numClients; i++){
//                listOfClients[i]->getLearner()->setMeanAction(mean_action);
//                listOfClients[i]->getLearner()->updateF(current_time, mean_action);
//
//
//        }
//
////        cout<<"Probability of action 1 in s1: "<<mean_prob_s1_a1<<endl;
////        cout<<"Probability of action 2 in s1: "<<mean_prob_s1_a2<<endl;
////
////        cout<<"Probability of action 1 in s2: "<<mean_prob_s2_a1<<endl;
////        cout<<"Probability of action 2 in s2: "<<mean_prob_s2_a2<<endl;
////
////        cout<<"Number of received tasks: "<<numRecievedPackets<<endl;
////        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;
////        cout<<"mean of action 1: "<<mean_action[0]<<endl;
////        cout<<"mean of action 2: "<<mean_action[1]<<endl;
//
//        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;
//        cout<<"Client 1, server 1 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[0]<<", Client 1, server 2 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[1]<<endl;
//        cout<<"Client 2, server 1 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[0]<<", Client 2, server 2 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[1]<<endl;
//        cout<<"Client 3, server 1 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[1]<<endl;
//
//
//        assert(mean_action[0] <= 1);
//        assert(mean_action[1] <= 1);
//
//
//        for(int i=0; i<numClients; i++){
//
//            if(learning_algorithm=="MFQ"){
//
//                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
//
//                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
//                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
//                    listOfClients[i]->getLearner()->resetGradientSum();
//                    listOfClients[i]->updateCounter();
//                    listOfClients[i]->resetBatchCounter();
//                }
//                else if(listOfClients[i]->getPacketReceived()) listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
//
//            }
//        }
//
//        // print results
//        if (current_time > currentInterval * samplingInterval) {
//            currentInterval++;
//            this->printResults();
//            this->saveResults(configIndex);
//
//        }
//    }
//}



void Simulator::printResults(){
    // do nothing if print level equals zero
    if(printLevel == 0){
        // nothing
    }
    // print servers info if print level equals one
    if(printLevel == 1){
        for (int i =0; i < numServers ; i++){
            auto avgServiceTime = listOfServers[i]->getAvgServiceTime();
            auto numDepartures = listOfServers[i]->getNumDepartures();
            auto avgWaitTime = listOfServers[i]->getAvgWaitTime();
            cout<<"Server "<<i+1<<"\n"<< "Average service time: "<<avgServiceTime<<", Number of departures: "<<numDepartures<<", Average wait time: "<<avgWaitTime<<endl;
        }
        
    }
    // print both servers and clients info if print level equals 2
    if(printLevel == 2){
        
        for (int i =0; i < numServers ; i++){
            
            double minProb = 1.0;
            double meanProb = 0.0;
            
            for (int j =0; j < numClients ; j++){
                
    //            auto avgLatency = listOfClients[i]->getAverageLatency();
    //            cout<<"Client "<<i+1<<"\n"<< "Average latency: "<<avgLatency<<endl;
                auto prob = listOfClients[j]->getLearner()->getProbabilities();
                
    //            cout<<"Client "<<i+1<<endl;
                
    //            for(int j=0; j<numServers; j++) cout<<"Server "<<j+1<<":"<< prob[j]<<endl;
                if (prob[i] < minProb) minProb = prob[i];
                
                meanProb += prob[i]/numClients;
            }
            
            auto avgServiceTime = listOfServers[i]->getAvgServiceTime();
            auto numDepartures = listOfServers[i]->getNumDepartures();
            auto avgWaitTime = listOfServers[i]->getAvgWaitTime();
            cout<<"Server "<<i+1<<"\n"<< "Average service time: "<<avgServiceTime<<", Number of departures: "<<numDepartures<<", Average wait time: "<<avgWaitTime<<"\n"<<"Average probability: "<<meanProb<< ", Minimum probability: "<< minProb <<endl;
        }
    }
}

void Simulator::saveResults(int configIndex){
    //save probabilities
    // load on servers - average wait time
    string config = "Config file";
    char config_index = configIndex + '0';
    string directory = "/Users/fatemehfardno/Desktop/PSGA/PSGA/Results/";
    string client = "Client";
//    string server = "Server";
//    string fileName3 = config + config_index + "-Average latency";
//    string fileName4 = config + config_index + "-Latency";
//    string fileName5 = config + config_index + "-Percentile";
//    ofstream file3;
//    file3.open(directory+fileName3, ios_base::app);
//    ofstream file4;
//    file4.open(directory+fileName4, ios_base::app);
//    ofstream file5;
//    file5.open(directory+fileName5, ios_base::app);
    for (int i=0; i<numClients; i++){
        char clientID = (i+1) + '0';
        auto reward = listOfClients[i]->getLearner()->getReward();
//        auto avgLatency = listOfClients[i]->getAverageLatency();
//        auto latency = listOfClients[i]->getlatency();
//        auto percentile = this->percentile(listOfClients[i]->getLatencyVector());
        string fileName1 = config + config_index + "-reward-"+ client + clientID;
        //string fileName2 = config + config_index + "-Servers load";
        ofstream file1;
        file1.open(directory+fileName1, ios_base::app);
//        ofstream file2;
//        file2.open(directory+fileName2, ios_base::app);
//        file3<<avgLatency<<" ";
//        file4<<latency<<" ";
//        file5<<percentile<<" ";
        file1<<reward<<" ";
        file1<<endl;
        file1.close();
//        file2<<endl;
//        file2.close();
  }
//    file3<<endl;
//    file3.close();
//    file4<<endl;
//    file4.close();
//    file5<<endl;
//    file5.close();
    
}


//TODO: two clients, check what happens for one
// if Q = v and if Q<0 => delta >0 => GS <0 => Q' > 0!!!!!
