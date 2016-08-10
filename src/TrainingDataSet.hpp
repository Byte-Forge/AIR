#pragma once
#include <vector>
#include <memory>
#include "NeuralNetwork.hpp"

class TrainingDataSet
{
public:
	std::vector<std::shared_ptr<DataEntry>> trainingSet;
	std::vector<std::shared_ptr<DataEntry>> generalizationSet;
	std::vector<std::shared_ptr<DataEntry>> validationSet;

	TrainingDataSet() {}

	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};