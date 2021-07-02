#pragma once

#include <fstream>
#include <iostream>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "tinyxml2.h"

using namespace std;
using namespace cv;
using namespace tinyxml2;

class Model {
public:
    dnn::Net net;
    vector<String> outNames;
    vector<string> classNamesVec;
    filesystem::path p;
    Mat img;

    void initModel();
    bool setImgPath(string);
    void detectObj(filesystem::path);
    bool writeXML();
};
