//
//  header.cpp
//  discrete_simulator
//
//  Created by Fatemeh Fardno on 2022-07-01.
//

#include "MFQ.hpp"

using namespace std;

ExecutionDistribution getExecutionDist(string a) {
    if (a == "CONSTANT")
        return CONSTANT;
    if ( a == "GEOM")
        return GEOM;
    exit(-1);
}

LearningAlgorithm getLearningAlgorithm(string a){
    if( a == "MFQ")
        return MFQ;
    exit(-1);
}


int MFQ::predict(){
    double exp_sum = 0.0;
    double sum = 0.0;
    double Q = 0.0;
    int index = 0;
    double probSum = 0.0;
    double max_Q = 0.0;

    for(int i=0; i<numOfChoices; i++){
        index += state[i]*(capacity - 1.0)*IntPower(capacity,i);
    }
    
    vector<double> temp1;
    for(int i=0; i<numOfChoices; i++){
        temp1.push_back(f[index][i]);
    }
    assert(temp1.size()==numOfChoices);

    
    for (int i=0; i<numOfChoices; i++){
        Q = 0.0;
        vector<double> temp = getBasis(i, state, temp1);
        assert(temp.size()==theta_size);
        for (int j=0; j<theta_size; j++){
            Q += theta[j] * temp[j];
                }
        if(i==0){
            max_Q = Q;
        }
        else if (Q >= max_Q) max_Q = Q;
    }
    
    for (int i=0; i<numOfChoices; i++){
        Q = 0.0;
        vector<double> temp = getBasis(i, state, temp1);
        
        for (int j=0; j<theta_size; j++){
            Q += theta[j] * temp[j];
        
        }
        
        probability[i] = exp(Beta * (Q - max_Q +100.0));
        exp_sum += probability[i];
    }
    
    for (int i=0; i<numOfChoices; i++){
        probability[i] = probability[i]/exp_sum;
        sum += probability[i];
    }

//    cout<<sum<<endl;
    assert(abs(1.0-sum) < 0.0000001);

    for(int i=0; i<numOfChoices; i++){
        probSum += (1 - disabled_server[i]) * probability[i];
    }
    
    auto val = (*distribution)(generator);
    int choice = 0;
    double var = 0.0;
    
    for (int i = 0; i < numOfChoices; i++){
        if(disabled_server[i] == 0){
            var = var + probability[i]/probSum;
            if (val <= var){
                choice = i;
                break;
        }
    }
}
    
    // make sure the choice is valid
    assert(choice < numOfChoices);
    assert(probSum > 0.0);
    assert(disabled_server[choice] == 0);
    return choice;
    
}

void MFQ::setState(vector<double> stateValue){
    previous_state = state;
    for(int i=0; i<numOfChoices; i++){
        state[i] = (stateValue[i]+1.0)/capacity;
    }
    assert(stateValue.size()==numOfChoices);
    assert(state.size()==numOfChoices);
}


void MFQ::setMeanAction(vector<double> meanAction) {
    mean_action = meanAction;
    assert(meanAction.size()==numOfChoices);
};

void MFQ::updateF(int time, vector<double> mean_action, int numPackets) {
    
    int index = 0;
    for(int i=0; i<numOfChoices; i++){
        index += (previous_state[i]*(capacity - 1.0))*IntPower(capacity,i);
    }
    
    double temp = 0.0;
    for(int i =0; i<numOfChoices; i++) temp += mean_action[i];
    
    if(temp > 0.0){
        for(int i=0; i<numOfChoices; i++){
            if(time == 0) f[index][i] = 2.0 * mean_action[i];
            else f[index][i] = (time * f[index][i] + 2.0 * mean_action[i]) / (time+1);
    }
  }
   
}



void MFQ::updateGradientSum(double reward, int choice){
    
    double delta = 0.0;
    double v = 0.0;
    int index = 0;
    int index_previous = 0;
    double Q_previous = 0.0;
    
    for(int i=0; i<numOfChoices; i++){
        index += (state[i])*(capacity - 1.0)*IntPower(capacity,i);
        index_previous += (previous_state[i])*(capacity - 1.0)*IntPower(capacity,i);
    }

    vector<double> temp3;
    for(int i=0; i<numOfChoices; i++){
        temp3.push_back(f[index][i]);
    }
    assert(temp3.size()==numOfChoices);
    
    vector<double> temp4;
    for(int i=0; i<numOfChoices; i++){
        temp4.push_back(f[index_previous][i]);
    }
    assert(temp4.size()==numOfChoices);
        
    double max_Q = 0.0;
    double exp_sum = 0.0;
    double sum = 0.0;
    vector<double> prob;
    prob.resize(numOfChoices, 0.0);
    
    for (int i=0; i<numOfChoices; i++){
        double Q = 0.0;
        vector<double> temp1 = getBasis(i, state, temp3);
        for (int j=0; j<theta_size; j++){
            Q += theta[j] * temp1[j];
        }
        
        if(i==0){
            max_Q = Q;
        }
        else if (Q >= max_Q) max_Q = Q;
    }
    
//    assert(max_Q <= 40.0/(1-discount_factor));
    
    for (int i=0; i<numOfChoices; i++){
        double Q = 0.0;
        vector<double> temp1 = getBasis(i, state, temp3);
        for (int j=0; j<theta_size; j++){
            Q += theta[j] * temp1[j];
        }
        prob[i] = exp(Beta * (Q - max_Q +100.0));
        exp_sum += prob[i];
    }
    
    for (int i=0; i<numOfChoices; i++){
        prob[i] = prob[i]/exp_sum;
        sum = sum + prob[i];
    }
    
    assert(sum-1.0<0.0000001);
    assert(prob.size()==numOfChoices);
    
    
    for (int i=0; i<numOfChoices; i++){
        double Q = 0.0;
        vector<double> temp1 = getBasis(i, state, temp3);
        for (int j=0; j<theta_size; j++){
            Q += theta[j] * temp1[j];
        }
        v += Q * prob[i];
    }
   
    vector<double> temp2 = getBasis(choice, previous_state, temp4);
    for (int j=0; j<theta_size; j++){
        Q_previous += theta[j] * temp2[j];
    }
    assert(temp2.size()==theta_size);

    delta = reward + discount_factor * v - Q_previous;
    //cout<< delta << "..." << reward << "... "<<v << ":" << Q_previous <<endl;
    
    if(Adam==false){
        for (int i=0; i<theta_size; i++){
            GradientSum[i] =  GradientSum[i]  - delta * (2.0/batchSize) *  temp2[i];
        }
    }
    
    else if (Adam==true){
        counter++;
        for(int i=0; i<theta_size; i++){
            mAdam[i] = momentumParameter * mAdam[i] + (1.0 - momentumParameter) * ((-1.0) * delta * (2.0/batchSize) *  temp2[i]);
            vAdam[i] = rmsParameter * vAdam[i] + (1.0 - rmsParameter) * pow(delta * (2.0/batchSize) *  temp2[i],2);
            theta[i] = theta[i] - 0.001 * mAdam[i]/(sqrt(vAdam[i])+pow(10,-8));

            
//            if (this->getID()==0) cout<<"adam" << i << ":::"<<mAdam[i]<<endl;
//            if (this->getID()==0) cout<<"theta" << i << ":::"<<theta[i]<<endl;
//            theta[i] = theta[i] - 0.001 * (mAdam[i]/(1.0 - pow(momentumParameter,counter)))/((sqrt(vAdam[i]/(1.0 - pow(rmsParameter,counter)))+pow(10,-8)));
        }
        I = I * discount_factor;
    }
}

double MFQ::getBatchSize(){
    return batchSize;
}

double MFQ::getReward(){
    return reward;
}

void MFQ::setReward(double reward_val){
    reward = reward_val;
}

bool MFQ::getAdam(){
    return Adam;
}

void MFQ::updateWeights(double reward, int choice){
//    cout<<"BATCH"<<endl;
//    counter++;
    if(Adam==true) {
        updateGradientSum(reward, choice);
        return;
    }
    
    updateGradientSum(reward, choice);
//
//    cout << "START" << endl;
//    for(int j=0; j<theta_size; j++){
//        cout << theta[j] << endl;
//    }
//    
//    cout << "!!##$$" << endl;
    
//    for(int j=0; j<theta_size; j++){
//        V_adam[j] = momentumParameter * V_adam[j] + (1.0 - momentumParameter) * GradientSum[j];
//        S_adam[j] = rmsParameter * S_adam[j] + (1.0 - rmsParameter) * pow(GradientSum[j],2);
//        theta[j] = theta[j] - I * (V_adam[j])/(sqrt(S_adam[j]) + pow(10,-8));
//    }
    
//
        for(int j=0; j<theta_size; j++){
            theta[j] = theta[j] - I * GradientSum[j];
        }

    
//    for(int j=0; j<theta_size; j++){
//        cout << theta[j] << endl;
//    }
//    cout << "**" << endl;


//    assert(V_adam.size()==theta_size);
//    assert(S_adam.size()==theta_size);
\

    I = I * discount_factor;
    if (I <= 0.001) I = 0.001;
    
}

void MFQ::resetGradientSum(){
    for (int j = 0; j<theta_size; j++){
        GradientSum[j] = 0.0;
        }
}

void MFQ::getDisabledServer(vector<int> disabled){
    disabled_server = disabled;
}

bool MFQ::getF(){
    return fAdded;
}

vector<double> MFQ::getProbabilities(){
    return probability;
}

vector<double> MFQ::getState(){
    return state;
}

int MFQ::getID(){
    return id;
}

vector<double> MFQ::getPreviousState(){
    return previous_state;
}

// output is a vector that should be multiplied by theta
vector<double> MFQ::getBasis(int choice, vector<double> state, vector<double> f){
    
    vector<double> action;
    action.resize(numOfChoices, 0.0);
    action[choice] = 1.0;
    
    assert(action.size()==numOfChoices);
            
    vector<double> output;
    
    if(fAdded==false){
        for(int i =0; i<numOfChoices; i++){
            for(int j=0; j<numOfChoices; j++){
                output.push_back(state[j] * action[i]);
            }

        }
        
    }
    
    else if(fAdded==true){
        for(int i =0; i<numOfChoices; i++){
            for(int j=0; j<numOfChoices; j++){
                output.push_back(state[j] * action[i]);
                for(int k=0; k<numOfChoices; k++){
                    output.push_back(state[j] * f[k] * action[i]);
                }
            }

        }
        
    }
    
    output.push_back(1.0);
    
    assert(output.size()==theta_size);

    return output;
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
    
    rewardBuffer.resize(numClients, 0.0);
    choiceBuffer.resize(numClients, 0.0);
    vector<double> default_vector;
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
    
    mean_action.resize(numServers, 0.0);
    for(int i=0; i<numServers; i++) mean_action[i] = 0.0;
        
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
        
        double arrival_rate = clientsArrivalRate[0]/numClients;
        arrival_prob.push_back(arrival_rate);
        arrival_prob.push_back(1.0 - arrival_rate);

        Learning * lr;
        generator.seed(chrono::system_clock::now().time_since_epoch().count());
        distribution = new uniform_real_distribution<double>(0.0, 1.0);
        switch (getLearningAlgorithm(learningAlgorithmVector[0])){
            case MFQ:
                learning_algorithm = "MFQ";
                lr = new class MFQ(numClients, i, numServers, Beta, batchSize, discount_factor, learningParameters[0], capacity, alpha_theta[0], momentParam, RMSParam);
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
//        cout << listOfServers[i]->getQueueSize() << endl;
    }
    
}

void Simulator::testRun1(int configIndex){
    
    int currentInterval = 0;
    
    for(current_time=0; current_time<simulationTime; current_time++){
        
        double mean_prob_s1_a1 = 0.0;
        double mean_prob_s1_a2 = 0.0;
        
        double mean_prob_s2_a1 = 0.0;
        double mean_prob_s2_a2 = 0.0;
        
        double state1_count = 0.0;
        double state2_count = 0.0;
        
        vector<double> mean_action_s1;
        mean_action_s1.resize(numServers, 0.0);
        vector<double> mean_action_s2;
        mean_action_s2.resize(numServers, 0.0);
        numRecievedPackets = 0;

        for(int i=0; i<numClients; i++){
            rewardBuffer[i] = 0.0;
            choiceBuffer[i] = 0;
        }
        
        for(int i=0; i<numServers; i++){
            mean_action_s1[i] = 0.0;
            mean_action_s2[i] = 0.0;

            mean_action[i] = 0.0;
        }
        
        for(int i =0; i<numClients; i++){
            
            Client * client = listOfClients[i];
            
            auto val = (*distribution)(generator);
            client->resetPacketReceived();
            //bool packetReceived = false;
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
                double reward = 0.0;
//                cout<<"state:"<<endl;
//                cout<<state[0]<<endl;
//                cout<<state[1]<<endl;

                if(client->getLearner()->getState()[0]==1.0 && client->getLearner()->getState()[1]==0.0){
                    mean_action_s1[serverID] += 1.0;
                    state1_count += 1.0;
                    if(serverID==0){
                        reward = 80.0;
                        client->getLearner()->setReward(reward);
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
                    if(serverID==1){
                        reward = 100.0;
                        client->getLearner()->setReward(reward);
                        state[0] = 0.0;
                        state[1] = 1.0;
                    }
                    mean_prob_s1_a1 += client->getLearner()->getProbabilities()[0];
                    mean_prob_s1_a2 += client->getLearner()->getProbabilities()[1];
                }

                else if(client->getLearner()->getState()[0]==0.0 && client->getLearner()->getState()[1]==1.0){
                    mean_action_s2[serverID] += 1.0;
                    state2_count += 1.0;
                    reward = 1.0;
                    client->getLearner()->setReward(reward);
                    auto val = (*distribution)(generator);
                    if (val <= 0.9){
                        state[0] = 0.0;
                        state[1] = 1.0;
                        
                    }
                    else{
                        state[0] = 1.0;
                        state[1] = 0.0;
                    }
                    mean_prob_s2_a1 += client->getLearner()->getProbabilities()[0];
                    mean_prob_s2_a2 += client->getLearner()->getProbabilities()[1];

                }
                
                //if(serverID == 2) reward = 1.0;
                client->getLearner()->setState(state);
                rewardBuffer[i] = reward;
                choiceBuffer[i] = serverID;
                
            }
            
            for(int k=0; k<numClients; k++){
                listOfClients[k]->getLearner()->getDisabledServer(disabled_servers);
            
            }

        }
//        cout<<state1_count<<"**"<<state2_count<<"%%"<<numRecievedPackets<<endl;
        if(state1_count != 0){
            mean_prob_s1_a1 = mean_prob_s1_a1/state1_count;
            mean_prob_s1_a2 = mean_prob_s1_a2/state1_count;
            for(int i=0;i<numServers; i++) mean_action_s1[i] = mean_action_s1[i]/state1_count;

        }

        if(state2_count != 0){

            mean_prob_s2_a1 = mean_prob_s2_a1/state2_count;
            mean_prob_s2_a2 = mean_prob_s2_a2/state2_count;
            for(int i=0;i<numServers; i++) mean_action_s2[i] = mean_action_s2[i]/state2_count;

        }
        vector<double> zero;
        zero.resize(numServers, 0.0);
        
        for(int i=0; i<numClients; i++){
            if(listOfClients[i]->getLearner()->getPreviousState()[0]==1){
                listOfClients[i]->getLearner()->setMeanAction(mean_action_s1);
                listOfClients[i]->getLearner()->updateF(current_time, mean_action_s1, numRecievedPackets);

            }
            if(listOfClients[i]->getLearner()->getPreviousState()[1]==1){
                listOfClients[i]->getLearner()->setMeanAction(mean_action_s2);
                listOfClients[i]->getLearner()->updateF(current_time, mean_action_s2, numRecievedPackets);

            }
        }
        
        cout<<"Probability of action 1 in s1: "<<mean_prob_s1_a1<<endl;
        cout<<"Probability of action 2 in s1: "<<mean_prob_s1_a2<<endl;

        cout<<"Probability of action 1 in s2: "<<mean_prob_s2_a1<<endl;
        cout<<"Probability of action 2 in s2: "<<mean_prob_s2_a2<<endl;
        
        
        for(int i=0; i<numClients; i++){

            if(learning_algorithm=="MFQ"){
                
                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
                
                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
                    
                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
                    listOfClients[i]->getLearner()->resetGradientSum();
                    listOfClients[i]->updateCounter();
                    listOfClients[i]->resetBatchCounter();
                }
                else if(listOfClients[i]->getPacketReceived()) listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
                
            }
        }
        
        // print results
        if (current_time > currentInterval * samplingInterval) {
            currentInterval++;
            this->printResults();
            this->saveResults(configIndex);
            
        }
    }
}

void Simulator::testRun2(int configIndex){
    
    int currentInterval = 0;
    state[0] = 0.0;
    state[1] = 0.1;

    for(int i=0; i<numClients; i++) listOfClients[i]->getLearner()->setState(state);

    state[0] = 1.0;
    state[1] = 0.0;


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
        
        for(int i=0; i<numServers; i++) 
            mean_action[i] = 0.0;

        
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
        
        for(int i=0;i<numServers; i++){
            if(numRecievedPackets != 0) mean_action[i] = mean_action[i]/numRecievedPackets;
        }



        
        for(int i=0; i<numClients; i++){
                listOfClients[i]->getLearner()->setMeanAction(mean_action);
                listOfClients[i]->getLearner()->updateF(current_time, mean_action, numRecievedPackets);
            

        }
        
//        cout<<"Probability of action 1 in s1: "<<mean_prob_s1_a1<<endl;
//        cout<<"Probability of action 2 in s1: "<<mean_prob_s1_a2<<endl;
//
//        cout<<"Probability of action 1 in s2: "<<mean_prob_s2_a1<<endl;
//        cout<<"Probability of action 2 in s2: "<<mean_prob_s2_a2<<endl;
//
//        cout<<"Number of received tasks: "<<numRecievedPackets<<endl;
//        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;
//        cout<<"mean of action 1: "<<mean_action[0]<<endl;
//        cout<<"mean of action 2: "<<mean_action[1]<<endl;
        
        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;
        cout<<"Client 1, server 1 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[0]<<", Client 1, server 2 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 2, server 1 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[0]<<", Client 2, server 2 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 3, server 1 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[1]<<endl;

        cout << "Reward" << listOfClients[0]->getLearner()->getReward() << endl;
        cout << "Reward" << listOfClients[1]->getLearner()->getReward() << endl;
        cout << "Reward" << listOfClients[2]->getLearner()->getReward() << endl;
        
//        cout<<"Client 4, server 1 prob: "<<listOfClients[3]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[3]->getLearner()->getProbabilities()[1]<<endl;
//        cout<<"Client 5, server 1 prob: "<<listOfClients[4]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[4]->getLearner()->getProbabilities()[1]<<endl;
//        cout<<"Client 6, server 1 prob: "<<listOfClients[5]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[5]->getLearner()->getProbabilities()[1]<<endl;
//        cout<<"Client 7, server 1 prob: "<<listOfClients[6]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[6]->getLearner()->getProbabilities()[1]<<endl;
        

        assert(mean_action[0] <= 1);
        assert(mean_action[1] <= 1);

        
        for(int i=0; i<numClients; i++){

            if(learning_algorithm=="MFQ"){
                
                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
                
                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
                    listOfClients[i]->getLearner()->resetGradientSum();
                    listOfClients[i]->updateCounter();
                    listOfClients[i]->resetBatchCounter();
                }
                else if(listOfClients[i]->getPacketReceived()) listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
                
            }
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
        
        for(int i=0; i<numServers; i++) mean_action[i] = 0.0;

        
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
                cout<<"id: "<<client->getId()<<endl;
                cout<<"exec: "<<client->getTaskExecTime()<<endl;
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
                cout << server->getId() << endl;
                server->emplaceTask(task);
                // client gets expected latency as cost
                choiceBuffer[i] = serverID;
                mean_action[serverID] += 1.0;
            
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
        
        for(int i=0;i<numServers; i++){
            if(numRecievedPackets != 0) mean_action[i] = mean_action[i]/numRecievedPackets;
        }

        
        for(int i=0; i<numClients; i++){
                listOfClients[i]->getLearner()->setMeanAction(mean_action);
                listOfClients[i]->getLearner()->updateF(current_time, mean_action, numRecievedPackets);
        
        }
        
        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;

        cout<<"Client 1, server 1 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[0]<<", Client 1, server 2 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 2, server 1 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[0]<<", Client 2, server 2 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 3, server 1 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[1]<<endl;
 

        assert(mean_action[0] <= 1);
        assert(mean_action[1] <= 1);

        
        for(int i=0; i<numClients; i++){

            if(learning_algorithm=="MFQ"){
                
                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
                
                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
                    listOfClients[i]->getLearner()->resetGradientSum();
                    listOfClients[i]->updateCounter();
                    listOfClients[i]->resetBatchCounter();
                }
                else if(listOfClients[i]->getPacketReceived()) listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
                
            }
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
        
        for(int i=0; i<numServers; i++) mean_action[i] = 0.0;

        
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
        
        for(int i=0;i<numServers; i++){
            if(numRecievedPackets != 0) mean_action[i] = mean_action[i]/numRecievedPackets;
        }



        
        for(int i=0; i<numClients; i++){
                listOfClients[i]->getLearner()->setMeanAction(mean_action);
                listOfClients[i]->getLearner()->updateF(current_time, mean_action, numRecievedPackets);
            

        }
        
        cout<<"State: "<<listOfClients[0]->getLearner()->getPreviousState()[0]<<" ,"<<listOfClients[0]->getLearner()->getPreviousState()[1]<<endl;
        cout<<"Client 1, server 1 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[0]<<", Client 1, server 2 prob: "<<listOfClients[0]->getLearner()->getProbabilities()[1]<<endl;
        cout<<"Client 2, server 1 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[0]<<", Client 2, server 2 prob: "<<listOfClients[1]->getLearner()->getProbabilities()[1]<<endl;
 
//        cout<<"Client 3, server 1 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[0]<<", Client 3, server 2 prob: "<<listOfClients[2]->getLearner()->getProbabilities()[1]<<endl;
        

        assert(mean_action[0] <= 1);
        assert(mean_action[1] <= 1);

        
        for(int i=0; i<numClients; i++){

            if(learning_algorithm=="MFQ"){
                
                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
                
                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
                    listOfClients[i]->getLearner()->resetGradientSum();
                    listOfClients[i]->updateCounter();
                    listOfClients[i]->resetBatchCounter();
                }
                else if(listOfClients[i]->getPacketReceived()) listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
                
            }
        }
        
        // print results
        if (current_time > currentInterval * samplingInterval) {
            currentInterval++;
            this->printResults();
            this->saveResults(configIndex);
            
        }
    }
}

void Simulator::discreteRun(int configIndex){
    
    int currentInterval = 0;
    for(current_time=0; current_time<simulationTime; current_time++){
//        cout<<"time"<<current_time<<endl;
        numRecievedPackets = 0;

        for(int i=0; i<numClients; i++){
            rewardBuffer[i] = 0.0;
            choiceBuffer[i] = 0;
        }
        
        for(int i=0; i<numServers; i++){
            
            mean_action[i] = 0.0;
            double rate = listOfServers[i]->getRate();
            
            while(rate != 0.0 && listOfServers[i]->getQueueSize() != 0){
                
                Task * top_task = listOfServers[i]->getTop();
                
                if(top_task->getExecutionTime() - rate > 0.0){
                    top_task->setExecTime(top_task->getExecutionTime() - rate);
                    rate = 0.0;
                }
                
                else if(top_task->getExecutionTime() - rate <= 0.0){
                    auto latency = current_time - top_task->getArrivalTime();
                    top_task->getClient()->setLatency(latency);
                    top_task->getClient()->updateLatencyVector(latency);
                    top_task->getClient()->updateAvgLatency(latency);
                    
                    listOfServers[i]->serveNextTask(current_time);
                    rate -= top_task->getExecutionTime();

                }
            }
            
            state[i] = listOfServers[i]->getQueueSize();
            assert(state[i] <= capacity-1);
            if(state[i] <= capacity - 2) disabled_servers[i] = 0;
            if(state[i] == capacity - 1 || state[i] > capacity - 1) disabled_servers[i]  = 1;
            
            for(int k=0; k<numClients; k++){
                listOfClients[k]->getLearner()->getDisabledServer(disabled_servers);
            }
        }
        
        for(int k=0; k<numClients; k++) listOfClients[k]->getLearner()->setState(state);
        
//        double reward = 0.0;
        for(int i =0; i<numClients; i++){
            
            Client * client = listOfClients[i];
            
            auto val = (*distribution)(generator);
            client->resetPacketReceived();
            //bool packetReceived = false;
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
                //update state
                state[serverID] = listOfServers[serverID]->getQueueSize();

                assert(state[serverID] <= capacity-1);
                // if hit the limit, disable the server
                if(state[serverID] == capacity - 1 || state[serverID] > capacity - 1 ){
                    disabled_servers[serverID]  = 1;
                }
                if(state[serverID] <= capacity - 2) disabled_servers[serverID] = 0;
                
                // client gets expected latency as cost
//                double reward = 1.0 / pow((server->getAvgWaitTime()+rewardEpsilon),2);
                
//                double reward1 = 0.0;
//                if(serverID==0) reward1 = 100.0;
//                else reward1 = 1.0;

                double reward1 = 1.0 / (pow(server->getExpectedWaitTime(),2) + 0.05);
//                assert(reward <= 20.0);
//                reward += 1.0 / (pow(server->getExpectedWaitTime(),2) + 0.05);
                client->getLearner()->setReward(reward1);
                rewardBuffer[i] = reward1;
                choiceBuffer[i] = serverID;
                
            }

            else if(!client->getPacketReceived()) client->getLearner()->setReward(0.0);
            
            for(int k=0; k<numClients; k++){
                listOfClients[k]->getLearner()->getDisabledServer(disabled_servers);
            }

        }
        
        for(int k =0; k<numClients; k++) listOfClients[k]->getLearner()->setState(state);
        
        for(int j=0; j<numServers; j++) {
            if(numRecievedPackets != 0){
                for(int i=0; i<numClients; i++){
                    if(choiceBuffer[i] == j && listOfClients[i]->getPacketReceived()){
                        mean_action[j] += 1.0/numRecievedPackets;
                    }
                }
            }
            else mean_action[j] = 0.0;
        }
        
        double mean_action_sum = 0.0;
        for(int i=0; i<numServers; i++) mean_action_sum += mean_action[i];
        assert(mean_action_sum-1.0 < 0.00000001);
        
        for(int i=0; i<numClients; i++){
            listOfClients[i]->getLearner()->setMeanAction(mean_action);
            listOfClients[i]->getLearner()->updateF(current_time, mean_action, numRecievedPackets);
            
            if(learning_algorithm=="MFQ"){
                
                auto batchSize = listOfClients[i]->getLearner()->getBatchSize();
                
                if (listOfClients[i]->getBatchCounter() > batchSize - 1 ){
                    listOfClients[i]->getLearner()->updateWeights(rewardBuffer[i], choiceBuffer[i]);
                    listOfClients[i]->getLearner()->resetGradientSum();
                    listOfClients[i]->updateCounter();
                    listOfClients[i]->resetBatchCounter();
                }
                else if(listOfClients[i]->getPacketReceived()) {
                    
                    listOfClients[i]->getLearner()->updateGradientSum(rewardBuffer[i], choiceBuffer[i]);
                }
                
            }
        }
        
        // print results
        if (current_time > currentInterval * samplingInterval) {
            currentInterval++;
            this->printResults();
            this->saveResults(configIndex);
            
        }
    }
    
}


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

                auto prob = listOfClients[j]->getLearner()->getProbabilities();

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
    string directory = "/Users/fatemehfardno/Desktop/discrete_simulator/discrete_simulator/Results/";
    string client = "Client";
        double reward = 0.0;
        for (int i=0; i<numClients; i++){
            if(listOfClients[i]->getLearner()->getReward()!=0.0) {
                reward += log(listOfClients[i]->getLearner()->getReward());
            }
      }
    
    string fileName1;
    if(listOfClients[0]->getLearner()->getF()==false) fileName1 = config + config_index + "-reward-"+"IQL";
    else if(listOfClients[0]->getLearner()->getF()==true) fileName1 = config + config_index + "-reward-"+"MFQ";
    ofstream file1;
    file1.open(directory+fileName1, ios_base::app);
    file1<<reward<<" ";
    file1<<endl;
    file1.close();
    
    string fileName2;
    fileName2 = "Server 0 avgWaitTime";
    if(listOfClients[0]->getLearner()->getF()==false) fileName2 = fileName2 + "-IQL";
    else if(listOfClients[0]->getLearner()->getF()==true) fileName2 = fileName2 + "-MFQ";
    ofstream file2;
    file2.open(directory+fileName2, ios_base::app);
    file2<<listOfServers[0]->getAvgWaitTime()<<" ";
    file2<<endl;
    file2.close();
    
    string fileName3;
    fileName3 = "Server 1 avgWaitTime";
    if(listOfClients[0]->getLearner()->getF()==false) fileName3 = fileName3 + "-IQL";
    else if(listOfClients[0]->getLearner()->getF()==true) fileName3 = fileName3 + "-MFQ";
    ofstream file3;
    file3.open(directory+fileName3, ios_base::app);
    file3<<listOfServers[1]->getAvgWaitTime()<<" ";
    file3<<endl;
    file3.close();
    
    string fileName4;
    fileName4 = "Server 2 avgWaitTime";
    if(listOfClients[0]->getLearner()->getF()==false) fileName4 = fileName4 + "-IQL";
    else if(listOfClients[0]->getLearner()->getF()==true) fileName4 = fileName4 + "-MFQ";
    ofstream file4;
    file4.open(directory+fileName4, ios_base::app);
    file4<<listOfServers[2]->getAvgWaitTime()<<" ";
    file4<<endl;
    file4.close();
}

