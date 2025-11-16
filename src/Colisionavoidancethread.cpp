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
      
      double left_vel, right_vel;
      neuralNetwork->forward(sonar_input, left_vel, right_vel);
      
      // Log raw predictions
      double raw_left = left_vel;
      double raw_right = right_vel;
      
      // Prevent spinning: constrain velocities to prevent excessive turning
      const double MAX_VEL_DIFF = 50.0;      // Maximum difference between left/right (mm/s) to prevent spinning
      const double MIN_VEL_THRESHOLD = 10.0; // Below this, treat as stopped
      
      // Ensure both velocities are positive (forward movement only)
      if (left_vel < 0) left_vel = 0;
      if (right_vel < 0) right_vel = 0;
      
      // Limit the difference between left and right to prevent spinning
      double vel_diff = right_vel - left_vel;
      if (std::abs(vel_diff) > MAX_VEL_DIFF) {
            // Clamp the difference to prevent excessive turning
            double avg_vel = (left_vel + right_vel) / 2.0;
            left_vel = avg_vel - MAX_VEL_DIFF / 2.0;
            right_vel = avg_vel + MAX_VEL_DIFF / 2.0;
            
            // Ensure still positive
            if (left_vel < 0) {
                  left_vel = 0;
                  right_vel = MAX_VEL_DIFF;
            }
            if (right_vel < 0) {
                  right_vel = 0;
                  left_vel = MAX_VEL_DIFF;
            }
      }
      
      // If both velocities are very low, stop completely (avoid slow drift)
      if (left_vel < MIN_VEL_THRESHOLD && right_vel < MIN_VEL_THRESHOLD) {
            left_vel = 0;
            right_vel = 0;
      }
      
      robo->Move(left_vel, right_vel);
      
      // Always log to debug - show raw predictions and constrained values
      std::cout << "NN Raw: L=" << raw_left << " R=" << raw_right 
                << " | Constrained: L=" << left_vel << " R=" << right_vel
                << " | Sonar[0]=" << sonar_input[0] << std::endl;
}
