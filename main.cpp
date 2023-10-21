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
    unet.addLayer(Layer("conv_0",1,3,64,258,3));
    unet.addLayer(Layer("conv_1",2,64,64,258,3));
    unet.addLayer(Layer("pool_2",3,64,64,258,3));
    /*unet.addLayer(Layer("conv_3",4,64,128,130),3);
    unet.addLayer(Layer("conv_4",5,128,128,130,3));
    unet.addLayer(Layer("pool_5",6,128,128,130,3));
    unet.addLayer(Layer("conv_6",7,128,256,66,3));
    unet.addLayer(Layer("conv_7",8,256,256,66,3));
    unet.addLayer(Layer("pool_8",9,256,256,66,3));
    unet.addLayer(Layer("conv_9",10,256,512,34,3));
    unet.addLayer(Layer("conv_10",11,512,512,34,3));
    unet.addLayer(Layer("pool_11",12,512,512,34,3));
    unet.addLayer(Layer("conv_12",13,512,1024,18,3));
    unet.addLayer(Layer("conv_13",14,1024,1024,18,3));
    unet.addLayer(Layer("upsample_14",15,1024,1024,18,3));
    unet.addLayer(Layer("conv_15",16,1024,512,34,3));
    unet.addLayer(Layer("conv_16",17,1024,512,34,3));
    unet.addLayer(Layer("conv_17",18,512,512,34,3));
    unet.addLayer(Layer("upsample_18",19,512,512,34,3));
    unet.addLayer(Layer("conv_19",20,512,256,66,3));
    unet.addLayer(Layer("conv_20",21,512,256,66,3));
    unet.addLayer(Layer("conv_21",22,256,256,66,3));
    unet.addLayer(Layer("upsample_22",23,256,256,66,3));
    unet.addLayer(Layer("conv_23",24,256,128,130,3));
    unet.addLayer(Layer("conv_24",25,256,128,130,3));
    unet.addLayer(Layer("conv_25",26,128,128,130,3));
    unet.addLayer(Layer("upsample_26",27,128,128,130,3));
    unet.addLayer(Layer("conv_27",28,128,64,258,3));
    unet.addLayer(Layer("conv_28",29,128,64,258,3));
    unet.addLayer(Layer("conv_29",30,64,64,258,3));
    unet.addLayer(Layer("conv_30",31,64,5,256,3));
    unet.addLayer(Layer("unet_31",32,5,5,256,3));*/
    unet.start();
    waitKey(0);
    return 0;
} 