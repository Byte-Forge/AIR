#pragma once
#include <vector>
#include <string>
#include <memory>
#include "DataEntry.hpp"
#include "TrainingDataSet.hpp"

namespace air
{
	//dataset retrieval approach enum
	enum { NONE, STATIC, GROWING, WINDOWING };

	class DataReader
	{
	public:
		DataReader();
		~DataReader();

		bool loadDataFile(const std::string& filename, int nI, int nT);
		void setCreationApproach(int approach, double param1 = -1, double param2 = -1);
		int getNumTrainingSets();

		std::shared_ptr<TrainingDataSet> getTrainingDataSet();
		std::vector<std::shared_ptr<DataEntry>>& getAllDataEntries();

	private:
		void createStaticDataSet();
		void createGrowingDataSet();
		void createWindowingDataSet();
		void processLine(std::string &line);

	private:

		//data storage
		std::vector<std::shared_ptr<DataEntry>> data;
		int nInputs;
		int nTargets;

		//current data set
		std::shared_ptr<TrainingDataSet> tSet;

		//data set creation approach and total number of dataSets
		int creationApproach;
		int numTrainingSets;
		int trainingDataEndIndex;

		//creation approach variables
		double growingStepSize;			//step size - percentage of total set
		int growingLastDataIndex;		//last index added to current dataSet
		int windowingSetSize;			//initial size of set
		int windowingStepSize;			//how many entries to move window by
		int windowingStartIndex;		//window start index	
	};
}

