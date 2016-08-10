#pragma once
#include "DataReader.hpp"
#include "DataEntry.hpp"

class NeuralNetwork
{
public:
	//constructor & destructor
	NeuralNetwork(int numInput, int numHidden, int numOutput);
	~NeuralNetwork();

	bool loadWeights(char* inputFilename);
	bool saveWeights(char* outputFilename);
	int* feedForwardPattern(double* pattern);
	double getSetAccuracy(std::vector<DataEntry*>& set);
	double getSetMSE(std::vector<DataEntry*>& set);
	int clampOutput(double x);
	void feedForward(double* pattern);

private:
	void initializeWeights();
	inline double activationFunction(double x);

public:
	//number of neurons
	int nInput, nHidden, nOutput;

	//neurons
	double* inputNeurons;
	double* hiddenNeurons;
	double* outputNeurons;

	//weights
	double** wInputHidden;
	double** wHiddenOutput;
};
