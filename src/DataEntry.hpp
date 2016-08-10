#pragma once
#include <iostream>
#include <vector>

using namespace std;

class DataEntry
{
public:
	double* pattern;	//all the patterns
	double* target;		//all the targets

public:
	DataEntry(double* p, double* t): pattern(p), target(t) {}
		
	~DataEntry()
	{				
		delete[] pattern;
		delete[] target;
	}
};
