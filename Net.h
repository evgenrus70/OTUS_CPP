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
        Mat image;
        int imgSize;

        Net (std::string,std::string,std::string,std::string,int);
        void print();
        void printImage(int);
        int readImage (std::string);
        int imageToInput(int);
        int readWeights (std::string);
        void addLayer(Layer); 
        void printLayers();
        void start();      
};