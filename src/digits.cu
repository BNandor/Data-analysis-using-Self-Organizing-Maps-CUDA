#include <iostream>
#include <limits.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include "SOM.cu"

using namespace std;

#define training_file "optdigits.tra"
#define test_file "optdigits.tes"

#ifndef outputSOM
#define outputSOM "som.txt"
#endif

#ifndef mapWidth
#define mapWidth 10
#endif

#ifndef mapHeight
#define mapHeight 10
#endif

#ifndef defaultFramecount
#define defaultFramecount 100
#endif

#ifndef defaultClasscount
#define defaultClasscount 10
#endif

bool classify(std::vector<SelfOrganizingMap> &maps, const digit &sample)
{
	double minDist = std::numeric_limits<double>::max();
	double d;
	int closestMapIndex = -1;
	int i = 0;
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

std::map<std::string, std::string> parse_options(int argc, const char *argv[])
{
	std::map<std::string, std::string> options;
	for (int i = 1; i < argc; i++)
	{
		int eq = std::string(argv[i]).find("=");
		if (eq != std::string::npos)
		{
			std::string option = std::string(argv[i]).substr(0, eq);
			if (strlen(argv[i]) > eq + 1)
			{
				std::string value = std::string(argv[i]).substr(eq + 1, strlen(argv[i]));
				options[option] = value;
			}
		}
	}
	return options;
}


int main(int argc, const char *argv[])
{

	std::map<std::string, std::string> options = parse_options(argc, argv);
	
	int digitW = options.count("imagew") > 0 ? std::atoi(options["imagew"].c_str()) : 8;
	int digitH = options.count("imageh") > 0 ? std::atoi(options["imageh"].c_str()) : 8;

	int mapW = options.count("mapw") > 0 ? std::atoi(options["mapw"].c_str()) : mapWidth;
	int mapH = options.count("maph") > 0 ? std::atoi(options["maph"].c_str()) : mapHeight;

	int maxT = options.count("gen") > 0?std::atoi(options["gen"].c_str()):3000;

	std::cout << "[SOM] image dimensions : " << digitW << " x " << digitH << std::endl;
	std::cout << "[SOM] map dimensions : " << mapW << " x " << mapH << std::endl;

	// temp
	std::string inputName = training_file;
	std::string testName = test_file;

	if (!options.count("input"))
	{
		std::cerr << "[SOM] please specify an input file with  input=file" << std::endl;
		exit(0);
	}else{
		inputName = options["input"];
	}

	bool classification = options.count("test") > 0;
	bool animation = options.count("animation") > 0 || options.count("animationPath") > 0;

	if (!classification)
	{
		fstream input;
		ofstream output;

		std::string outputName = options.count("outputPath") > 0 ? options["outputPath"] : outputSOM;
		std::string animationPath = options.count("animationPath") > 0 ? options["animationPath"] : "./";
		int frameCount = options.count("framecount") > 0 ? std::atoi(options["framecount"].c_str()) : defaultFramecount;
		input.open(inputName.c_str());

		std::cout << "[SOM] reading input from " << inputName << std::endl;

		if (!input.is_open())
		{
			cerr << "could not open file" << inputName << endl;
			return -1;
		}

		digitSet inputSet(digitW, digitH);
		input >> inputSet;

		SelfOrganizingMap som(inputSet, mapW, mapH);
		if (animation)
		{
			std::cout<<"[SOM] animation enabled current framecount is "<<frameCount<<std::endl;
			std::cout<<"[SOM] saving animation to  "<<animationPath<<std::endl;

			som.train(maxT, [&](int T, int maxT, SelfOrganizingMap *map) {
				if ((T + maxT / frameCount) % (maxT / frameCount) == 0)
				{
					std::ofstream file;
					file.open(animationPath + (std::to_string(T / (maxT / frameCount)) + ".som").c_str());
					map->printMapToStream(file);
					file.close();
				}
			});
		}
		else
		{
				som.train(maxT);
		}
		std::cout<<"[SOM] saved final SOM to   "<<outputName<<std::endl;

		output.open(outputName);
		som.printMapToStream(output);
	}
	else
	{
		if (!options.count("test"))
		{
			std::cerr << "[SOM] please specify a test file with  test=file" << std::endl;
			exit(0);
		}
		else
		{
			testName = options["test"];
		}
		std::cout << "[SOM] classification mode on " << inputName << std::endl;

		fstream input, testinput;
		input.open(inputName);
		testinput.open(testName);
		std::cout << "[SOM] reading training data from " << inputName << std::endl;


		if (!input.is_open())
		{
			cerr << "could not open file" << inputName << endl;
			return -1;
		}

		std::cout << "[SOM] reading test data from " << testName << std::endl;
		if (!testinput.is_open())
		{
			cerr << "could not open file" << testName << endl;
			return -1;
		}

		int classCount = options.count("classCount") > 0 ? std::atoi(options["classCount"].c_str()) : defaultClasscount;
		std::cout << "[SOM] class count is " << classCount << std::endl;

		digitSet train(digitW, digitH);
		input >> train;

		digitSet filtered[classCount];
		for (int i = 0; i < classCount; i++)
		{
			filtered[i] = filterByValue(train, i);
		}

		digitSet test;
		testinput >> test;
		std::vector<SelfOrganizingMap> maps;

		for (int i = 0; i < classCount; i++)
		{
			maps.push_back(SelfOrganizingMap(filtered[i], mapW, mapH));
			maps[i].train(maxT);
		}

		std::cout << "[SOM] precision:" << std::count_if(test.getDigits().begin(), test.getDigits().end(), [&](auto &sample) {
										 return classify(maps, sample);
									 }) / (float)test.getDigits().size()
				  << std::endl;
	}
	return 0;
}
