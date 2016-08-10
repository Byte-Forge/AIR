#pragma once
#include <vector>
#include <string>
#include "DataEntry.hpp"
#include "TrainingDataSet.hpp"

//dataset retrieval approach enum
enum { NONE, STATIC, GROWING, WINDOWING };

class DataReader
{
public:
	DataReader(): creationApproach(NONE), numTrainingSets(-1) {}
	~DataReader();
	
	bool loadDataFile( const char* filename, int nI, int nT );
	void setCreationApproach( int approach, double param1 = -1, double param2 = -1 );
	int getNumTrainingSets();	
	
	TrainingDataSet* getTrainingDataSet();
	std::vector<DataEntry*>& getAllDataEntries();

private:
	void createStaticDataSet();
	void createGrowingDataSet();
	void createWindowingDataSet();	
	void processLine( std::string &line );

private:

	//data storage
	std::vector<DataEntry*> data;
	int nInputs;
	int nTargets;

	//current data set
	TrainingDataSet tSet;

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
