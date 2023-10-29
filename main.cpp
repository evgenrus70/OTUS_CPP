#include <iostream>
#include <opencv2/opencv.hpp>
#include "Net.h"

using namespace cv;

#define imgPath   "C://Users//Evgen//Desktop//OTUS_CPP//image//in_256.png"
#define wghtsPath "C://Users//Evgen//Desktop//OTUS_CPP//weights"

int main () {
    std::cout <<"Start U-Net CNN" <<std::endl;
    Net unet = Net("unet",wghtsPath,imgPath,"path/mask",256);
    unet.print();
    unet.readImage(unet.imagePath);
    unet.readWeights(unet.weightsPath);
    // Layer::Layer (std::string _type, int _numLayer, int _inFm, int _outFm, int _inSize, int _pad, int _coreSize, int _stride)
    unet.addLayer(Layer("conv",0,0,3,64,258,1,3,1));
    unet.addLayer(Layer("conv",1,0,64,64,258,1,3,1));
    unet.addLayer(Layer("pool",2,1,64,64,258,1,2,2));
    unet.addLayer(Layer("conv",3,2,64,128,130,1,3,1));
    unet.addLayer(Layer("conv",4,3,128,128,130,1,3,1));
    unet.addLayer(Layer("pool",5,4,128,128,130,1,2,2));
    unet.addLayer(Layer("conv",6,5,128,256,66,1,3,1));
    unet.addLayer(Layer("conv",7,6,256,256,66,1,3,1));
    unet.addLayer(Layer("pool",8,7,256,256,66,1,2,2));
    unet.addLayer(Layer("conv",9,8,256,512,34,1,3,1));
    unet.addLayer(Layer("conv",10,9,512,512,34,1,3,1));
    unet.addLayer(Layer("pool",11,10,512,512,34,1,2,2));
    unet.addLayer(Layer("conv",12,11,512,1024,18,1,3,1));
    unet.addLayer(Layer("conv",13,12,1024,1024,18,1,3,1));
    unet.addLayer(Layer("upsample",14,13,1024,1024,18,1,2,2));
    unet.addLayer(Layer("conv",15,14,1024,512,34,1,3,1));
    unet.addLayer(Layer("conv",16,10,1024,512,34,1,3,1));
    unet.addLayer(Layer("conv",17,16,512,512,34,1,3,1));
    unet.addLayer(Layer("upsample",19,17,512,512,34,1,2,2));
    unet.addLayer(Layer("conv",19,18,512,256,66,1,3,1));
    unet.addLayer(Layer("conv",20,7,512,256,66,1,3,1));
    unet.addLayer(Layer("conv",21,20,256,256,66,1,3,1));
    unet.addLayer(Layer("upsample",22,21,256,256,66,1,2,2));
    unet.addLayer(Layer("conv",23,22,256,128,130,1,3,1));
    unet.addLayer(Layer("conv",24,4,256,128,130,1,3,1));
    unet.addLayer(Layer("conv",25,24,128,128,130,1,3,1));
    unet.addLayer(Layer("upsample",26,25,128,128,130,1,2,2));
    unet.addLayer(Layer("conv",27,26,128,64,258,1,3,1));
    unet.addLayer(Layer("conv",28,1,128,64,258,1,3,1));
    unet.addLayer(Layer("conv",29,28,64,64,258,1,3,1));
    unet.addLayer(Layer("conv",30,29,64,5,256,1,3,1));
    unet.addLayer(Layer("last",31,30,5,5,256,1,3,1));
    unet.start();
    waitKey(0);
    return 0;
} 