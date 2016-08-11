#pragma once
#include <vector>

namespace air
{
	class DataEntry
	{
	public:
		DataEntry(std::vector<double> p, std::vector<double> t) : pattern(p), target(t)
		{

		}

	public:
		std::vector<double> pattern;	//all the patterns
		std::vector<double> target;		//all the targets
	};
}
