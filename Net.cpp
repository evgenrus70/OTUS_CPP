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
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            printf("%02hhx", image.at<Vec3b>(i, j)[0]);
            printf("%02hhx", image.at<Vec3b>(i, j)[1]);
            printf("%02hhx", image.at<Vec3b>(i, j)[2]);  
        }
        std::cout << std::endl;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
} 

int Net::readImage (std::string path) {
    std::cout <<"Image path: " << path <<std::endl;
    Mat image;
    image = imread(path, IMREAD_COLOR);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    printImage(image); 
    return 0;
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
