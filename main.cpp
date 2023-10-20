#include <iostream>
#include <opencv2/opencv.hpp>
#include "Net.h"

using namespace cv;

#define imgPath   "C://Users//evgenvt//Desktop//OTUS_CPP//image//in_256.png"
#define wghtsPath "C://Users//evgenvt//Desktop//OTUS_CPP//weights"

int main () {
    std::cout <<"Start U-Net CNN" <<std::endl;
    Net unet = Net("unet",wghtsPath,imgPath,"path/mask");
    unet.print();
    unet.readImage(unet.imagePath);
    unet.readWeights(unet.weightsPath);
    unet.addLayer(Layer("conv_0"));
    unet.addLayer(Layer("conv_1"));
    unet.addLayer(Layer("pool_2"));
    unet.addLayer(Layer("conv_3"));
    unet.addLayer(Layer("conv_4"));
    unet.printLayers();
    waitKey(0);
    return 0;
} 