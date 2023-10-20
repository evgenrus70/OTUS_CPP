#include <iostream>
#include <vector>
#include "Layer.h"
#include <opencv2/opencv.hpp>

using namespace cv;

class Net {
    public: 
        std::string name;
        std::string weightsPath;
        std::string imagePath;    
        std::string maskPath;
        std::vector<Layer> layers;

        Net (std::string,std::string,std::string,std::string);
        void print();
        void printImage(Mat);
        int readImage (std::string);
        int readWeights (std::string);
        void addLayer(Layer); 
        void printLayers();      
};