//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <algorithm>

#include "NeuralNetwork.hpp"

using namespace air;

NeuralNetwork::NeuralNetwork(int nI, int nH, int layers, int nO) : nInput(nI), nHidden(nH), m_layers(layers), nOutput(nO)
{
	//TODO: create layers of hidden neurons
	inputNeurons = std::vector<double>(nInput + 1, 0.0);
	/*inputNeurons.resize(nInput + 1);
	std::fill(inputNeurons.begin(), inputNeurons.end(), 0.0);*/

	//create input bias neuron
	inputNeurons[nInput] = -1;

	hiddenNeurons = std::vector<double>(nHidden + 1, 0.0);
	//create hidden bias neuron
	hiddenNeurons[nHidden] = -1;

	outputNeurons = std::vector<double>(nOutput + 1, 0.0);
	wInputHidden = std::vector<std::vector<double>>(nInput + 1, std::vector<double>(nHidden, 0.0));
	wHiddenOutput = std::vector<std::vector<double>>(nHidden + 1, std::vector<double>(nOutput, 0.0));

	initializeWeights();
}

NeuralNetwork::~NeuralNetwork()
{

}

bool NeuralNetwork::loadWeights(const std::string& filename)
{
	std::fstream inputFile;
	inputFile.open(filename, std::ios::in);

	if (inputFile.is_open())
	{
		std::vector<double> weights;
		std::string line = "";

		while (!inputFile.eof())
		{
			getline(inputFile, line);

			if (line.length() > 2)
			{
				char* cstr = new char[line.size() + 1];
				char* t;
				strcpy(cstr, line.c_str());

				//tokenise
				int i = 0;
				t = strtok(cstr, ",");

				while (t != NULL)
				{
					weights.push_back(atof(t));

					//move token onwards
					t = strtok(NULL, ",");
					i++;
				}

				//free memory
				delete[] cstr;
			}
		}

		//check if sufficient weights were loaded
		if (weights.size() != ((nInput + 1) * nHidden + (nHidden + 1) * nOutput))
		{
			std::cout << std::endl << "Error - Incorrect number of weights in input file: " << filename << std::endl;

			//close file
			inputFile.close();

			return false;
		}
		else
		{
			//set weights
			int pos = 0;

			for (int i = 0; i <= nInput; i++)
			{
				for (int j = 0; j < nHidden; j++)
				{
					wInputHidden[i][j] = weights[pos++];
				}
			}

			for (int i = 0; i <= nHidden; i++)
			{
				for (int j = 0; j < nOutput; j++)
				{
					wHiddenOutput[i][j] = weights[pos++];
				}
			}

			//print success
			std::cout << std::endl << "Neuron weights loaded successfuly from '" << filename << "'" << std::endl;

			//close file
			inputFile.close();

			return true;
		}
	}
	else
	{
		std::cout << std::endl << "Error - Weight input file '" << filename << "' could not be opened: " << std::endl;
		return false;
	}
}

bool NeuralNetwork::saveWeights(const std::string& filename)
{
	//open file for reading
	std::fstream outputFile;
	outputFile.open(filename, std::ios::out);

	if (outputFile.is_open())
	{
		outputFile.precision(50);

		//output weights
		for (int i = 0; i <= nInput; i++)
		{
			for (int j = 0; j < nHidden; j++)
			{
				outputFile << wInputHidden[i][j] << ",";
			}
		}

		for (int i = 0; i <= nHidden; i++)
		{
			for (int j = 0; j < nOutput; j++)
			{
				outputFile << wHiddenOutput[i][j];
				if (i * nOutput + j + 1 != (nHidden + 1) * nOutput) outputFile << ",";
			}
		}

		//print success
		std::cout << std::endl << "Neuron weights saved to '" << filename << "'" << std::endl;

		//close file
		outputFile.close();

		return true;
	}
	else
	{
		std::cout << std::endl << "Error - Weight output file '" << filename << "' could not be created: " << std::endl;
		return false;
	}
}

std::vector<int> NeuralNetwork::feedForwardPattern(std::vector<double> pattern)
{
	feedForward(pattern);

	//create copy of output results
	std::vector<int> results(nOutput);
	for (int i = 0; i < nOutput; i++) results[i] = clampOutput(outputNeurons[i]);

	return results;
}

double NeuralNetwork::getSetAccuracy(std::vector<std::shared_ptr<DataEntry>>& set)
{
	double incorrectResults = 0;

	//for every training input array
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(set[tp]->pattern);

		//correct pattern flag
		bool correctResult = true;

		//check all outputs against desired output values
		for (int k = 0; k < nOutput; k++)
		{
			//set flag to false if desired and output differ
			if (clampOutput(outputNeurons[k]) != set[tp]->target[k]) correctResult = false;
		}

		//inc training error for a incorrect result
		if (!correctResult) incorrectResults++;
	}

	 //calculate error and return as percentage
	return 100 - (incorrectResults / set.size() * 100);
}

double NeuralNetwork::getSetMSE(std::vector<std::shared_ptr<DataEntry>>& set)
{
	double mse = 0;

	//for every training input array
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(set[tp]->pattern);

		//check all outputs against desired output values
		for (int k = 0; k < nOutput; k++)
		{
			//sum all the MSEs together
			mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
		}
	}
	 //calculate error and return as percentage
	return mse / (nOutput * set.size());
}

void NeuralNetwork::initializeWeights()
{
	//set range
	double rH = 1 / sqrt((double)nInput);
	double rO = 1 / sqrt((double)nHidden);

	//set weights between input and hidden 		
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= nInput; i++)
	{
		for (int j = 0; j < nHidden; j++)
		{
			//set weights to random values
			wInputHidden[i][j] = (((double)(rand() % 100) + 1) / 100 * 2 * rH) - rH;
		}
	}

	//set weights between input and hidden
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= nHidden; i++)
	{
		for (int j = 0; j < nOutput; j++)
		{
			//set weights to random values
			wHiddenOutput[i][j] = (((double)(rand() % 100) + 1) / 100 * 2 * rO) - rO;
		}
	}

}

inline double NeuralNetwork::activationFunction(double x)
{
	//sigmoid function
	return 1 / (1 + exp(-x));
}

int NeuralNetwork::clampOutput(double x)
{
	if (x < 0.1) return 0;
	else if (x > 0.9) return 1;
	else return -1;
}

void NeuralNetwork::feedForward(std::vector<double> pattern)
{
	for (int i = 0; i < nInput; i++) inputNeurons[i] = pattern[i];

	//Calculate Hidden Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < nHidden; j++)
	{
		//clear value
		hiddenNeurons[j] = 0;

		//get weighted sum of pattern and bias neuron
		for (int i = 0; i <= nInput; i++) hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];

		//set to result of sigmoid
		hiddenNeurons[j] = activationFunction(hiddenNeurons[j]);
	}

	//Calculating Output Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < nOutput; k++)
	{
		//clear value
		outputNeurons[k] = 0;

		//get weighted sum of pattern and bias neuron
		for (int j = 0; j <= nHidden; j++) outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];

		//set to result of sigmoid
		outputNeurons[k] = activationFunction(outputNeurons[k]);
	}
}

