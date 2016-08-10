#pragma once
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include "NeuralNetwork.hpp"

//Constant Defaults!
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 90  
#define DESIRED_MSE 0.001 

/*******************************************************************
* Basic Gradient Descent Trainer with Momentum and Batch Learning
********************************************************************/
namespace air
{
	class NeuralNetworkTrainer
	{
	public:
		NeuralNetworkTrainer(std::shared_ptr<NeuralNetwork> untrainedNetwork);
		void setTrainingParameters(double lR, double m, bool batch);
		void setStoppingConditions(int mEpochs, double dAccuracy);
		void useBatchLearning(bool flag) { useBatch = flag; }
		void enableLogging(const std::string& filename, int resolution);

		void trainNetwork(std::shared_ptr<TrainingDataSet> tSet);

		//private methods
		//--------------------------------------------------------------------------------------------
	private:
		inline double getOutputErrorGradient(double desiredValue, double outputValue);
		double getHiddenErrorGradient(int j);
		void runTrainingEpoch(std::vector<std::shared_ptr<DataEntry>> trainingSet);
		void backpropagate(std::vector<double> desiredOutputs);
		void updateWeights();

	private:
		std::shared_ptr<NeuralNetwork> NN;

		//learning parameters
		double learningRate;					// adjusts the step size of the weight update	
		double momentum;						// improves performance of stochastic learning (don't use for batch)

												//epoch counter
		long epoch;
		long maxEpochs;

		//accuracy/MSE required
		double desiredAccuracy;

		//change to weights
		std::vector<std::vector<double>> deltaInputHidden;
		std::vector<std::vector<double>> deltaHiddenOutput;

		//error gradients
		std::vector<double> hiddenErrorGradients;
		std::vector<double> outputErrorGradients;

		//accuracy stats per epoch
		double trainingSetAccuracy;
		double validationSetAccuracy;
		double generalizationSetAccuracy;
		double trainingSetMSE;
		double validationSetMSE;
		double generalizationSetMSE;

		//batch learning flag
		bool useBatch;

		//log file handle
		bool loggingEnabled;
		std::fstream logFile;
		int logResolution;
		int lastEpochLogged;
	};
}

