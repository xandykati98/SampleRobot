// Quick test to check actual sensor configuration
// Compile and run with the robot to see sensor layout

#include "ClassRobo.h"
#include "Aria.h"
#include <iostream>

int main()
{
    int sucesso;
    PioneerRobot *robo = new PioneerRobot(ConexaoSimulacao, "", &sucesso);
    
    if (!sucesso) {
        std::cout << "Failed to connect" << std::endl;
        return 1;
    }
    
    ArUtil::sleep(1000);
    
    std::cout << "Robot Sonar Configuration:" << std::endl;
    std::cout << "Total sonar sensors: " << robo->robot.getNumSonar() << std::endl;
    
    for (int i = 0; i < robo->robot.getNumSonar() && i < 16; i++) {
        double x = robo->robot.getSonarX(i);
        double y = robo->robot.getSonarY(i);
        double angle = robo->robot.getSonarTh(i);
        int range = robo->robot.getSonarRange(i);
        
        std::cout << "Sensor " << i << ": ";
        std::cout << "Position (" << x << ", " << y << ") ";
        std::cout << "Angle: " << angle << "° ";
        std::cout << "Range: " << range << "mm" << std::endl;
    }
    
    Aria::exit(0);
    return 0;
}


