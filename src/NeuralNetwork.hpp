#pragma once
#include "DataReader.hpp"
#include <string>
#include <memory>

namespace air
{
	struct DataEntry
	{
		std::vector<double> pattern;	//all the patterns
		std::vector<double> target;		//all the targets
	};

	class NeuralNetwork
	{
	public:
		//constructor & destructor
		NeuralNetwork(int numInput, int numHidden, int layers, int numOutput);
		~NeuralNetwork();

		bool loadWeights(const std::string& inputFilename);
		bool saveWeights(const std::string& outputFilename);
		std::vector<int> feedForwardPattern(std::vector<double> pattern);
		double getSetAccuracy(std::vector<std::shared_ptr<DataEntry>>& set);
		double getSetMSE(std::vector<std::shared_ptr<DataEntry>>& set);
		int clampOutput(double x);
		void feedForward(std::vector<double> pattern);

	private:
		void initializeWeights();
		inline double activationFunction(double x);

	public:
		//number of neurons
		int nInput, nHidden, nOutput;
		int m_layers; //number of hidden neuron layers

					  //neurons
		std::vector<double> inputNeurons;
		std::vector<double> hiddenNeurons;
		std::vector<double> outputNeurons;

		//weights
		std::vector<std::vector<double>> wInputHidden;
		std::vector<std::vector<double>> wHiddenOutput;
	};

}
