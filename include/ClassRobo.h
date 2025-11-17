#ifndef CLASSROBO_H
#define CLASSROBO_H

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "Aria.h"

#define GirarBase 1
#define ConexaoSerial 1
#define ConexaoRadio 2
#define ConexaoSimulacao 3

class PioneerRobot
{
public:
  ArRobot robot;
  ArSonarDevice sonarDev;
  ArSick sick;

  ArLaserConnector *laserConnector;
  ArRobotConnector *robotConnector;
  ArArgumentParser *parser;
  ArSimpleConnector *simpleConnector;
  ArAnalogGyro *gyro;
  ArSerialConnection con1;

  int Sensores[8];
  PioneerRobot(int tipoConexao, const char *info, int *sucesso);

  void destroy();
  void desconectar();
  void pararMovimento();
  void readSensores();

  int getSonar(int i);
  int isConnected();

  float getXPos();
  float getYPos();
  float getAngBase();

  void initMov();
  void Rotaciona(double degrees, int Sentido, int velocidade);
  void getAllSonar(int *sensores);
  void Move(double vl, double vr);
  void setVel(int velocidade);
  void getLaser();
  void getWriteLaserReadings();
  void RunExit();
};

#endif
