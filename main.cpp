#include <iostream>



class Net {
    public: 
        std::string name;
        std::string weightsPath;
        std::string inputPath;    
        std::string maskPath;

        Net (std::string name,std::string weightsPath,std::string inputPath,std::string maskPath) {
            
        }      
};


int main () {
    std::cout <<"Start U-Net CNN" <<std::endl;
    return 0;
} 