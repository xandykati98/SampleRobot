#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

class NeuralNetwork
{
private:
    std::vector<int> architecture;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    double max_velocity;

    std::vector<double> tanh_activation(const std::vector<double> &input);
    std::vector<double> matmul(const std::vector<double> &input, 
                                const std::vector<std::vector<double>> &weight_matrix,
                                const std::vector<double> &bias_vector);

public:
    NeuralNetwork();
    bool loadFromJson(const std::string &filepath);
    void forward(const std::vector<double> &input, double &left_vel, double &right_vel);
    bool isLoaded() const;
};

#endif // NEURALNETWORK_H

