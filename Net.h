#include <iostream>
#include <vector>
#include "Layer.h"

class Net {
    public: 
        std::string name;
        std::string weightsPath;
        std::string inputPath;    
        std::string maskPath;
        std::vector<Layer> layers;

        Net (std::string,std::string,std::string,std::string);
        void print();
        void openImage (std::string path);
        void addLayer(Layer); 
        void printLayers();      
};