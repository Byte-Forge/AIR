#pragma once
#include <vector>
#include "DataEntry.hpp"

class TrainingDataSet
{
public:

	std::vector<DataEntry*> trainingSet;
	std::vector<DataEntry*> generalizationSet;
	std::vector<DataEntry*> validationSet;

	TrainingDataSet() {}

	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};