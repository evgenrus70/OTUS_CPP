#include <iostream>
#include <opencv2/opencv.hpp>
#include "Net.h"

using namespace cv;

#define imgPath   "C://Users//Evgen//Desktop//OTUS_CPP//image//in_256.png"
#define wghtsPath "C://Users//Evgen//Desktop//OTUS_CPP//weights"

int main () {
    std::cout <<"Start U-Net CNN" <<std::endl;
    Net unet = Net("unet",wghtsPath,imgPath,"path/mask");
    unet.print();
    unet.readImage(unet.imagePath);
    unet.readWeights(unet.weightsPath);
    unet.addLayer(Layer("conv_0",1,3,64,256));
    unet.addLayer(Layer("conv_1",2,3,64,256));
    unet.addLayer(Layer("pool_2",3,3,64,256));
    unet.addLayer(Layer("conv_3",4,3,64,256));
    unet.addLayer(Layer("conv_4",5,3,64,256));
    unet.printLayers();
    unet.start();
    waitKey(0);
    return 0;
} 