#include "ClassRobo.h"
#include "Aria.h"
#include <iostream>
#include "Config.h"
#include "Colisionavoidancethread.h"
#include "Wallfollowerthread.h"
#include "Sonarthread.h"
#include "Laserthread.h"
#include "NeuralNetwork.h"

PioneerRobot *robo;

int main(int argc, char **argv)
{
    int sucesso;

    robo = new PioneerRobot(ConexaoSimulacao, "", &sucesso);

    // Load neural network if available
    NeuralNetwork *neuralNetwork = new NeuralNetwork();
    bool nnLoaded = neuralNetwork->loadFromJson("weights.json");
    
    if (!nnLoaded)
    {
        ArLog::log(ArLog::Normal, "Neural network not loaded, will use rule-based control");
        delete neuralNetwork;
        neuralNetwork = nullptr;
    }

    ArLog::log(ArLog::Normal, "Criando as theads...");
    ColisionAvoidanceThread colisionAvoidanceThread(robo, neuralNetwork);
    // WallFollowerThread wallFollowerThread(robo);
    SonarThread sonarReadingThread(robo);
    // LaserThread laserReadingThread(robo);

    ArLog::log(ArLog::Normal, "Sonar Readings thread ...");
    sonarReadingThread.runAsync();

    // ArLog::log(ArLog::Normal, "Laser Readings thread ...");
    // laserReadingThread.runAsync();

    ArLog::log(ArLog::Normal, "Colision Avoidance thread ...");
    colisionAvoidanceThread.runAsync();

    // ArLog::log(ArLog::Normal, "Wall Following thread ...");
    // wallFollowerThread.runAsync();

    robo->robot.waitForRunExit();

    if (neuralNetwork != nullptr)
    {
        delete neuralNetwork;
    }

    Aria::exit(0);
}
