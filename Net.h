#include <iostream>

class Net {
    public: 
        std::string name;
        std::string weightsPath;
        std::string inputPath;    
        std::string maskPath;

        Net (std::string,std::string,std::string,std::string);
        void print();      
};