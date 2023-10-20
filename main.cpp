#include <iostream>
#include "Net.h"

int main () {
    std::cout <<"Start U-Net CNN" <<std::endl;
    Net unet = Net("unet","path/weights","path/inputs","path/mask");
    unet.print();
    unet.addLayer(Layer("conv_0"));
    unet.addLayer(Layer("conv_1"));
    unet.addLayer(Layer("pool_2"));
    unet.addLayer(Layer("conv_3"));
    unet.addLayer(Layer("conv_4"));
    unet.printLayers();
    return 0;
} 