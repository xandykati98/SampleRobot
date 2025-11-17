#include "Colisionavoidancethread.h"
#include "Config.h"
#include <iostream>
#include <vector>
#include <cmath>

ColisionAvoidanceThread::ColisionAvoidanceThread(PioneerRobot *_robo, NeuralNetwork *nn)
{
      this->robo = _robo;
      this->neuralNetwork = nn;
      this->useNeuralNetwork = (nn != nullptr && nn->isLoaded());
      this->last_action = 0;  // Initialize to forward action
      
      if (this->useNeuralNetwork)
      {
            std::cout << "Using Neural Network for collision avoidance" << std::endl;
      }
      else
      {
            std::cout << "Using rule-based collision avoidance" << std::endl;
      }
}

void *ColisionAvoidanceThread::runThread(void *)
{
      while (this->getRunningWithLock())
      {
            myMutex.lock();
            robo->getAllSonar(sonar);
            
            if (useNeuralNetwork)
            {
                  tratamentoNeuralNetwork();
            }
            else
            {
                  tratamentoSimples();
            }
            
            // ArUtil::sleep(1000);
            myMutex.unlock();
      }

      ArLog::log(ArLog::Normal, "Colision Avoidance.");
      return NULL;
}

void ColisionAvoidanceThread::waitOnCondition() { myCondition.wait(); }

void ColisionAvoidanceThread::lockMutex() { myMutex.lock(); }

void ColisionAvoidanceThread::unlockMutex() { myMutex.unlock(); }

void ColisionAvoidanceThread::tratamentoSimples()
{
      int sumD = (sonar[3] * LIMIARFRENTE) + ((sonar[2] + sonar[1]) * LIMIARDIAGONAIS) + (sonar[0] * LIMIARLATERAIS); // 2
      int sumE = (sonar[4] * LIMIARFRENTE) + ((sonar[5] + sonar[6]) * LIMIARDIAGONAIS) + (sonar[7] * LIMIARLATERAIS); // 1
      int dirMov = 1;

      if (robo->robot.isHeadingDone())
      {
            std::cout << "A ultima rotacao foi concluida \n";
            if (sumD > sumE)
                  dirMov = 2;
            if (sonar[3] <= LIMIARFRENTE / 5 || sonar[4] <= LIMIARFRENTE / 5)
            {
                  robo->Move(-VELOCIDADEDESLOCAMENTO, -VELOCIDADEDESLOCAMENTO);
                  std::cout << "Frente perto \n";
            }
            else if (sonar[0] <= LIMIARLATERAIS)
            {
                  robo->Rotaciona(5, dirMov, VELOCIDADEROTACAO);
                  std::cout << "Esquerda perto \n";
            }
            else if (sonar[1] <= LIMIARDIAGONAIS || sonar[2] <= LIMIARDIAGONAIS)
            {
                  robo->Rotaciona(15, dirMov, VELOCIDADEROTACAO);
                  std::cout << "DDE perto \n";
            }
            else if (sonar[3] <= LIMIARFRENTE || sonar[4] <= LIMIARFRENTE)
            {
                  robo->Rotaciona(45, dirMov, VELOCIDADEROTACAO);
                  std::cout << "Frente afastado \n";
            }
            else if (sonar[5] <= LIMIARDIAGONAIS || sonar[6] <= LIMIARDIAGONAIS)
            {
                  robo->Rotaciona(15, dirMov, VELOCIDADEROTACAO);
                  std::cout << "DDD perto \n";
            }
            else if (sonar[7] <= LIMIARLATERAIS)
            {
                  robo->Rotaciona(5, dirMov, VELOCIDADEROTACAO);
                  std::cout << "Direita perto \n";
            }
            else
            {
                  robo->Move(VELOCIDADEDESLOCAMENTO, VELOCIDADEDESLOCAMENTO);
                  std::cout << "Seguir em frente \n";
            }
      }
      else
            std::cout << "Executando rotacao previa \n";
}

void ColisionAvoidanceThread::tratamentoNeuralNetwork()
{
      std::vector<double> sonar_input(8);
      for (int i = 0; i < 8; i++)
      {
            sonar_input[i] = static_cast<double>(sonar[i]);
      }
      
      // Get action from neural network (0=forward, 1=correct_left, 2=correct_right)
      int action = neuralNetwork->getAction(sonar_input, last_action);
      
      // Store action for next iteration
      last_action = action;
      
      // Apply discrete action
      const double FORWARD_VEL = 150.0;  // Base forward velocity (mm/s)
      const double TURN_VEL_DIFF = 50.0;  // Velocity difference for turning
      
      double left_vel, right_vel;
      
      if (action == 0) {  // Forward
            left_vel = FORWARD_VEL;
            right_vel = FORWARD_VEL;
            std::cout << "Action: FORWARD";
      }
      else if (action == 1) {  // Correct left
            left_vel = FORWARD_VEL - TURN_VEL_DIFF;
            right_vel = FORWARD_VEL + TURN_VEL_DIFF;
            std::cout << "Action: CORRECT_LEFT";
      }
      else if (action == 2) {  // Correct right
            left_vel = FORWARD_VEL + TURN_VEL_DIFF;
            right_vel = FORWARD_VEL - TURN_VEL_DIFF;
            std::cout << "Action: CORRECT_RIGHT";
      }
      else {  // Default to forward
            left_vel = FORWARD_VEL;
            right_vel = FORWARD_VEL;
            std::cout << "Action: DEFAULT_FORWARD";
      }
      
      robo->Move(left_vel, right_vel);
      
      // Log action and velocities
      std::cout << " | L=" << left_vel << " R=" << right_vel
                << " | [0]=" << sonar_input[0]
                << " | [1]=" << sonar_input[1]
                << " | [2]=" << sonar_input[2]
                << " | [3]=" << sonar_input[3]
                << " | [4]=" << sonar_input[4] << std::endl;
}
