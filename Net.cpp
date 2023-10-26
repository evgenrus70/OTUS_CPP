#include "Net.h"

Net::Net (std::string _name,std::string _weightsPath,std::string _imagePath,std::string _maskPath, int _imgSize) {
    name = _name;
    weightsPath = _weightsPath;
    imagePath = _imagePath;
    maskPath = _maskPath;
    imgSize = _imgSize;
}
void Net::print(){
    
}

void Net::printImage(int pad){
    int imgSizePad = imgSize + 2 * pad;
    int k = 0;
    for (int i = 0; i < imgSizePad; i++) {
        for (int j = 0; j < imgSizePad; j++) {
            printf("%f ", layers[0].inputData[imgSizePad * imgSizePad * k + imgSizePad * j + i]); // x_size * y_size * z + x_size * y + x
        }
        std::cout << std::endl;
    }
    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    //imshow("Display Image", image);
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

int Net::imageToInput (int pad) {
    int channelsCount = 3;
    int imgSizePad = imgSize + 2 * pad;
    int x = 0, y = 0;
    layers[0].inputData = new float[imgSizePad*imgSizePad*channelsCount];
    for (int i = 0; i < imgSizePad; i++) {
            for (int j = 0; j < imgSizePad; j++) {
                for (int k = 0; k < channelsCount; k++) {
                    layers[0].inputData[imgSizePad * imgSizePad * k + imgSizePad * j + i]  = 0;                                 
                }
            }
    }

    for (int i = 0; i < imgSizePad; i++) {
        if ((i >= pad) && (i <= imgSize)) {
            for (int j = 0; j < imgSizePad; j++) {
                if ((j >= pad) && (j <= imgSize)) {
                    for (int k = 0; k < channelsCount; k++) {
                        layers[0].inputData[imgSizePad * imgSizePad * k + imgSizePad * j + i]  = image.at<Vec3b>(x,y)[k];                                 
                    }
                    y++;
                }
            }
            x++;
            y=0;
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
    imageToInput(1);
    printImage(1);
    //layers[0].forward();
    int i = 0;
    for (auto& layer : layers) {
        //layer.forward();
        //if (i < layers.size() - 1)
        //    layers[i+1].inputData = layers[i].outputData;
        i++;
    }
    for (auto& layer : layers) {
        //layer.printInputs();
        //layer.printOutputs();
    }
}
