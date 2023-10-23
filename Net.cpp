#include "Net.h"

Net::Net (std::string _name,std::string _weightsPath,std::string _imagePath,std::string _maskPath) {
    name = _name;
    weightsPath = _weightsPath;
    imagePath = _imagePath;
    maskPath = _maskPath;
}
void Net::print(){
    
}

void Net::printImage(Mat image){
    //for (int i = 0; i < 256; i++) {
    //    for (int j = 0; j < 256; j++) {
            printf("%02hhx", image.at<Vec3b>(0, 0)[0]);
            printf("%02hhx", image.at<Vec3b>(0, 0)[1]);
            printf("%02hhx", image.at<Vec3b>(0, 0)[2]);  
    //    }
        std::cout << std::endl;
    //}
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
} 

int Net::readImage (std::string path) {
    std::cout <<"Image path: " << path <<std::endl;
    image = imread(path, IMREAD_COLOR);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
   // printImage(image); 
    return 0;
}

int Net::loadImage () {
    for (int i = 1; i < 257; i++) {
        for (int j = 1; j < 257; j++) {
            for (int k = 0; k < 3; k++) {
                layers[0].inputData(i,j,k) = image.at<Vec3b>(i-1,j-1)[k];
            }
        }
    }
    return 1;
}

int Net::readWeights (std::string path) {
     std::cout <<"Weights path: " << path <<std::endl;
     return 0;
}

void Net::addLayer (Layer layer) {
    layers.push_back(layer);
}

void Net::printLayers () {
    for (const auto& layer : layers) {
        std::cout << layer.name << std::endl;
    }
}

void Net::start () {
    loadImage ();
    //layers[0].forward();
    int i = 0;
    for (auto& layer : layers) {
        layer.forward();
        if (i < layers.size() - 1)
            layers[i+1].inputData = layers[i].outputData;
        i++;
    }
    for (auto& layer : layers) {
        //layer.printInputs();
        //layer.printOutputs();
    }
}
