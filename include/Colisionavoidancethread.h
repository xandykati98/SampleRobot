#ifndef COLISIONAVOIDANCETHREAD_H
#define COLISIONAVOIDANCETHREAD_H
#include "Aria.h"
#include "ClassRobo.h"
#include "NeuralNetwork.h"

class ColisionAvoidanceThread : public ArASyncTask
{
public:
    PioneerRobot *robo;
    ArCondition myCondition;
    ArMutex myMutex;
    int sonar[8];
    NeuralNetwork *neuralNetwork;
    bool useNeuralNetwork;
    int last_action;  // Track last action (0=forward, 1=left, 2=right)

public:
    ColisionAvoidanceThread(PioneerRobot *_robo, NeuralNetwork *nn);
    void *runThread(void *);
    void waitOnCondition();
    void lockMutex();
    void unlockMutex();
    void tratamentoSimples();
    void tratamentoNeuralNetwork();
};

#endif // COLISIONAVOIDANCETHREAD_H
