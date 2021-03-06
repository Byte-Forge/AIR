#include "DataReader.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <algorithm>
#include "NeuralNetwork.hpp"

using namespace air;

DataReader::DataReader() : creationApproach(NONE), numTrainingSets(-1) 
{
	tSet = std::make_shared<TrainingDataSet>();
}

/*******************************************************************
* Destructor
********************************************************************/
DataReader::~DataReader()
{

}

/*******************************************************************
* Loads a csv file of input data
********************************************************************/
bool DataReader::loadDataFile( const std::string& filename, int nI, int nT )
{
	//clear any previous data		
	data.clear();
	if (tSet != nullptr)
		tSet->clear();
	
	//set number of inputs and outputs
	nInputs = nI;
	nTargets = nT;

	//open file for reading
	std::fstream inputFile;
	inputFile.open(filename, std::ios::in);

	if ( inputFile.is_open() )
	{
		std::string line = "";
		
		//read data
		while ( !inputFile.eof() )
		{
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) processLine(line);
		}		
		
		//shuffle data
		random_shuffle(data.begin(), data.end());

		//split data set
		trainingDataEndIndex = (int) ( 0.6 * data.size() );
		int gSize = (int) ( ceil(0.2 * data.size()) );
		int vSize = (int) ( data.size() - trainingDataEndIndex - gSize );
							
		//generalization set
		for ( int i = trainingDataEndIndex; i < trainingDataEndIndex + gSize; i++ ) tSet->generalizationSet.push_back( data[i] );
				
		//validation set
		for ( int i = trainingDataEndIndex + gSize; i < (int) data.size(); i++ ) tSet->validationSet.push_back( data[i] );
		
		//print success
		std::cout << "Input File: " << filename << "\nRead Complete: " << data.size() << " Patterns Loaded"  << std::endl;

		//close file
		inputFile.close();
		
		return true;
	}
	else 
	{
		std::cout << "Error Opening Input File: " << filename << std::endl;
		return false;
	}
}

/*******************************************************************
* Processes a single line from the data file
********************************************************************/
void DataReader::processLine(std::string &line )
{
	//create new pattern and target
	std::vector<double> pattern = std::vector<double>(nInputs);
	std::vector<double> target = std::vector<double>(nTargets);
	
	//store inputs		
	char* cstr = new char[line.size()+1];
	char* t;
	strcpy(cstr, line.c_str());

	//tokenise
	int i = 0;
	t=strtok (cstr,",");
	
	while ( t!=NULL && i < (nInputs + nTargets) )
	{	
		if ( i < nInputs ) pattern[i] = atof(t);
		else target[i - nInputs] = atof(t);

		//move token onwards
		t = strtok(NULL,",");
		i++;			
	}
	
	/*cout << "pattern: ";
	for (int i=0; i < nInputs; i++) 
	{
		cout << pattern[i] << ",";
	}
	
	cout << " target: ";
	for (int i=0; i < nTargets; i++) 
	{
		cout << target[i] << " ";
	}
	cout << endl;*/


	//add to records
	data.push_back(std::make_shared<DataEntry>(pattern, target));		
}
/*******************************************************************
* Selects the data set creation approach
********************************************************************/
void DataReader::setCreationApproach( int approach, double param1, double param2 )
{
	//static
	if ( approach == STATIC )
	{
		creationApproach = STATIC;
		
		//only 1 data set
		numTrainingSets = 1;
	}

	//growing
	else if ( approach == GROWING )
	{			
		if ( param1 <= 100.0 && param1 > 0)
		{
			creationApproach = GROWING;
		
			//step size
			growingStepSize = param1 / 100;
			growingLastDataIndex = 0;

			//number of sets
			numTrainingSets = (int) ceil( 1 / growingStepSize );				
		}
	}

	//windowing
	else if ( approach == WINDOWING )
	{
		//if initial size smaller than total entries and step size smaller than set size
		if ( param1 < data.size() && param2 <= param1)
		{
			creationApproach = WINDOWING;
			
			//params
			windowingSetSize = (int) param1;
			windowingStepSize = (int) param2;
			windowingStartIndex = 0;			

			//number of sets
			numTrainingSets = (int) ceil( (double) ( trainingDataEndIndex - windowingSetSize ) / windowingStepSize ) + 1;
		}			
	}

}

/*******************************************************************
* Returns number of data sets created by creation approach
********************************************************************/
int DataReader::getNumTrainingSets()
{
	return numTrainingSets;
}
/*******************************************************************
* Get data set created by creation approach
********************************************************************/
std::shared_ptr<TrainingDataSet> DataReader::getTrainingDataSet()
{		
	switch ( creationApproach )
	{	
		case STATIC : createStaticDataSet(); break;
		case GROWING : createGrowingDataSet(); break;
		case WINDOWING : createWindowingDataSet(); break;
	}
	return tSet;
}
/*******************************************************************
* Get all data entries loaded
********************************************************************/
std::vector<std::shared_ptr<DataEntry>>& DataReader::getAllDataEntries()
{
	return data;
}

/*******************************************************************
* Create a static data set (all the entries)
********************************************************************/
void DataReader::createStaticDataSet()
{
	//training set
	for ( int i = 0; i < trainingDataEndIndex; i++ ) tSet->trainingSet.push_back( data[i] );		
}
/*******************************************************************
* Create a growing data set (contains only a percentage of entries
* and slowly grows till it contains all entries)
********************************************************************/
void DataReader::createGrowingDataSet()
{
	//increase data set by step percentage
	growingLastDataIndex += (int) ceil( growingStepSize * trainingDataEndIndex );		
	if ( growingLastDataIndex > (int) trainingDataEndIndex ) growingLastDataIndex = trainingDataEndIndex;

	//clear sets
	tSet->trainingSet.clear();
	
	//training set
	for ( int i = 0; i < growingLastDataIndex; i++ ) tSet->trainingSet.push_back( data[i] );			
}
/*******************************************************************
* Create a windowed data set ( creates a window over a part of the data
* set and moves it along until it reaches the end of the date set )
********************************************************************/
void DataReader::createWindowingDataSet()
{
	//create end point
	int endIndex = windowingStartIndex + windowingSetSize;
	if ( endIndex > trainingDataEndIndex ) endIndex = trainingDataEndIndex;		

	//clear sets
	tSet->trainingSet.clear();
					
	//training set
	for ( int i = windowingStartIndex; i < endIndex; i++ ) tSet->trainingSet.push_back( data[i] );
			
	//increase start index
	windowingStartIndex += windowingStepSize;
}

