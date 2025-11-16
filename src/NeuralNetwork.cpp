#include "NeuralNetwork.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

using json = nlohmann::json;

NeuralNetwork::NeuralNetwork()
{
    max_velocity = 300.0;  // Default value
}

bool NeuralNetwork::loadFromJson(const std::string &filepath)
{
    try
    {
        std::ifstream file(filepath);
        if (!file.is_open())
        {
            std::cerr << "Failed to open neural network weights file: " << filepath << std::endl;
            return false;
        }

        json j;
        file >> j;

        // Load architecture
        architecture = j["architecture"].get<std::vector<int>>();

        // Load max_velocity if present, otherwise use default
        if (j.contains("max_velocity"))
        {
            max_velocity = j["max_velocity"].get<double>();
        }
        else
        {
            max_velocity = 300.0;  // Default
        }

        // Load layers
        auto layers = j["layers"];
        weights.clear();
        biases.clear();

        for (const auto &layer : layers)
        {
            // Load weights (2D matrix)
            auto weight_matrix = layer["weights"].get<std::vector<std::vector<double>>>();
            weights.push_back(weight_matrix);

            // Load biases (1D vector)
            auto bias_vector = layer["biases"].get<std::vector<double>>();
            biases.push_back(bias_vector);
        }

        std::cout << "Neural network loaded successfully" << std::endl;
        std::cout << "Architecture: ";
        for (size_t i = 0; i < architecture.size(); i++)
        {
            std::cout << architecture[i];
            if (i < architecture.size() - 1)
                std::cout << " -> ";
        }
        std::cout << std::endl;
        std::cout << "Max velocity: " << max_velocity << std::endl;

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading neural network: " << e.what() << std::endl;
        return false;
    }
}

std::vector<double> NeuralNetwork::tanh_activation(const std::vector<double> &input)
{
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        output[i] = std::tanh(input[i]);
    }
    return output;
}

std::vector<double> NeuralNetwork::matmul(const std::vector<double> &input,
                                          const std::vector<std::vector<double>> &weight_matrix,
                                          const std::vector<double> &bias_vector)
{
    size_t output_size = weight_matrix.size();
    std::vector<double> output(output_size, 0.0);

    // Matrix multiplication: output = weight_matrix * input + bias
    for (size_t i = 0; i < output_size; i++)
    {
        double sum = 0.0;
        for (size_t j = 0; j < input.size(); j++)
        {
            sum += weight_matrix[i][j] * input[j];
        }
        output[i] = sum + bias_vector[i];
    }

    return output;
}

void NeuralNetwork::forward(const std::vector<double> &input, double &left_vel, double &right_vel)
{
    if (weights.empty() || biases.empty())
    {
        std::cerr << "Neural network not loaded!" << std::endl;
        left_vel = 0.0;
        right_vel = 0.0;
        return;
    }

    // Normalize input (sonar readings from 0-5000 to 0-1)
    std::vector<double> normalized_input(input.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        normalized_input[i] = input[i] / 5000.0;
    }

    // Layer 1: input -> hidden (with Tanh)
    std::vector<double> hidden1 = matmul(normalized_input, weights[0], biases[0]);
    hidden1 = tanh_activation(hidden1);

    // Layer 2: hidden -> hidden (with Tanh)
    std::vector<double> hidden2 = matmul(hidden1, weights[1], biases[1]);
    hidden2 = tanh_activation(hidden2);

    // Layer 3: hidden -> output (with Tanh and scaling)
    std::vector<double> output = matmul(hidden2, weights[2], biases[2]);
    output = tanh_activation(output);
    
    // Scale output from [-1, 1] to [-max_velocity, max_velocity]
    left_vel = output[0] * max_velocity;
    right_vel = output[1] * max_velocity;
}

bool NeuralNetwork::isLoaded() const
{
    return !weights.empty() && !biases.empty();
}

