#include "digits.cu"
#include <fstream>

int main(int argc,const char* argv[]) {
    std::string configFilename = "configfiles/classify_8x8.conf";
    std::ifstream config(configFilename);

    if(!config.is_open() ) {
        std::cerr<<"Could not open "<<configFilename<<std::endl;
        exit(1);
    }

    executeWithOptions(io::parse_options_configfile(config));
    return 0;
}
