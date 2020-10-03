#include <iostream>
#include <limits.h>
#include <fstream>
#include <vector>
#include "ocr.h"
#include <algorithm>

using namespace std;
#define training_file "optdigits.tra"
#define test_file "optdigits.tes"

int main(int argc, const char *argv[])
{
	fstream input,testinput;
	ofstream output;
	input.open(training_file);	
	testinput.open(test_file);	
	output.open("som.txt");
	if(!input.is_open()){
		cerr<<	"could not open file"<<training_file<<endl;
		return -1;
	}
	if(!testinput.is_open()){
		cerr<<	"could not open file"<<test_file<<endl;
		return -1;
	}
	digitSet train;
	input>>train;
	
	digitSet test;
	testinput>>test;
	
	SelfOrganizingMap som(train,10,10);	

	som.train();
	som.printMapToStream(output);
	som.printMap();
	output.close();
	return 0;

}

