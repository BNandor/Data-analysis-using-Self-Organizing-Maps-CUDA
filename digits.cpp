#include <iostream>
#include <limits.h>
#include <fstream>
#include <vector>
#include "ocr.h"
#include <algorithm>

using namespace std;
#define training_file "optdigits.tra"
#define test_file "optdigits.tes"

bool classify(std::vector<SelfOrganizingMap> &maps,const digit &sample)
{
	double minDist = std::numeric_limits<double>::max();
	double d;
	int closestMapIndex=-1;
	int i=0;
	std::for_each(maps.begin(), maps.end(), [&](auto &map) {
		d = map.getClosestPrototypeDistance(sample);

		if (d < minDist)
		{
			minDist = d;
			closestMapIndex = i;
		}
		i++;
	});
	return sample.getValue() == closestMapIndex;
}
int main(int argc, const char *argv[])
{
	fstream input, testinput;
	ofstream output;
	input.open(training_file);
	testinput.open(test_file);
	output.open("som.txt");
	if (!input.is_open())
	{
		cerr << "could not open file" << training_file << endl;
		return -1;
	}
	if (!testinput.is_open())
	{
		cerr << "could not open file" << test_file << endl;
		return -1;
	}

	digitSet train(8, 8);
	input >> train;

	digitSet filtered[10];
	for (int i = 0; i < 10; i++)
	{
		filtered[i] = filterByValue(train, i);
	}
	digitSet test;
	testinput >> test;
	std::vector<SelfOrganizingMap> maps;

	// SelfOrganizingMap som(filtered[0],40,40);
	// som.train();
	// som.printMapToStream(output);

	for (int i = 0; i < 10; i++)
	{
		maps.push_back(SelfOrganizingMap(filtered[i], 3, 3));
		maps[i].train();
	}
	std::cout << "precision:" << std::count_if(test.getDigits().begin(), test.getDigits().end(), 
								[&](auto &sample) {
									 return classify(maps,sample);
								 }) 
								 / (float)test.getDigits().size()
			  << std::endl;
	//som.printMap();
	//output.close();
	return 0;
}
