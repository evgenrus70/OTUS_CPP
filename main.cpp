#include <iostream>
#include "Net.h"

int main () {
    std::cout <<"Start U-Net CNN" <<std::endl;
    Net unet = Net("unet","path/weights","path/inputs","path/mask");
    unet.print();
    return 0;
} 