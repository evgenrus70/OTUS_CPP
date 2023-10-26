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
    unet.addLayer(Layer("conv",0,3,64,258,1,3,1));
    unet.addLayer(Layer("conv",1,64,64,258,1,3,1));
    unet.addLayer(Layer("pool",2,64,64,258,1,2,2));
    unet.addLayer(Layer("conv",3,64,128,130,1,3,1));
    unet.addLayer(Layer("conv",4,128,128,130,1,3,1));
    unet.addLayer(Layer("pool",5,128,128,130,1,2,2));
    unet.addLayer(Layer("conv",6,128,256,66,1,3,1));
    unet.addLayer(Layer("conv",7,256,256,66,1,3,1));
    unet.addLayer(Layer("pool",8,256,256,66,1,2,2));
    unet.addLayer(Layer("conv",9,256,512,34,1,3,1));
    unet.addLayer(Layer("conv",10,512,512,34,1,3,1));
    unet.addLayer(Layer("pool",11,512,512,34,1,2,2));
    unet.addLayer(Layer("conv",12,512,1024,18,1,3,1));
    unet.addLayer(Layer("conv",13,1024,1024,18,1,3,1));
    unet.addLayer(Layer("upsample",14,1024,1024,18,1,2,2));
    unet.addLayer(Layer("conv",15,1024,512,34,1,3,1));
    unet.addLayer(Layer("conv",16,1024,512,34,1,3,1));
    unet.addLayer(Layer("conv",17,512,512,34,1,3,1));
    unet.addLayer(Layer("upsample",19,512,512,34,1,2,2));
    unet.addLayer(Layer("conv",19,512,256,66,1,3,1));
    unet.addLayer(Layer("conv",20,512,256,66,1,3,1));
    unet.addLayer(Layer("conv",21,256,256,66,1,3,1));
    unet.addLayer(Layer("upsample",23,256,256,66,1,2,2));
    unet.addLayer(Layer("conv",23,256,128,130,1,3,1));
    unet.addLayer(Layer("conv",24,256,128,130,1,3,1));
    unet.addLayer(Layer("conv",25,128,128,130,1,3,1));
    unet.addLayer(Layer("upsample",27,128,128,130,1,2,2));
    unet.addLayer(Layer("conv",27,128,64,258,1,3,1));
    unet.addLayer(Layer("conv",28,128,64,258,1,3,1));
    unet.addLayer(Layer("conv",29,64,64,258,1,3,1));
    unet.addLayer(Layer("conv",30,64,5,256,1,3,1));
    unet.addLayer(Layer("last",31,5,5,256,1,3,1));
    unet.start();
    waitKey(0);
    return 0;
} 