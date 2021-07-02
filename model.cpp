#include "model.h"
//#define _DEBUG_

void Model::initModel()
{
    filesystem::path p("../../..");
    string ps = filesystem::absolute(p).string();
    cout << ps << endl;
    net = dnn::readNetFromDarknet(ps + "\\asset\\yolov4-tiny-obj-rc.cfg", ps + "\\asset\\yolov4-tiny-obj-rc_best.weights");
    net.setPreferableBackend(dnn::Backend::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(dnn::Target::DNN_TARGET_CPU);
    outNames = net.getUnconnectedOutLayersNames();
#ifdef _DEBUG_
    for (int i = 0; i < outNames.size(); i++)
        printf("output layer name : %s\n", outNames[i].c_str());
#endif // _DEBUG_

    ifstream classNamesFile(ps + "\\asset\\obj-rc.names");
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
        {
            classNamesVec.push_back(className);
#ifdef _DEBUG_
            printf("class name : %s\n", className.c_str());
#endif // _DEBUG_
        }
    }
}

bool Model::setImgPath(string path)
{
    p = path;
    return exists(p);
}

void Model::detectObj(filesystem::path p)
{
    Mat frame = imread(filesystem::absolute(p).string());
    vector<Mat> outs;
    vector<Rect> boxes;
    vector<int> classIds;
    vector<float> confidences;

    try {

        Mat inputBlob = cv::dnn::blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
        net.setInput(inputBlob);
        net.forward(outs, outNames);

        for (size_t i = 0; i < outs.size(); ++i)
        {
            // detected objects and C is a number of classes + 4 where the first 4
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > 0.5)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.1, 0.3, indices);
        if (0 != indices.size())
        {
            XMLDocument doc;
            doc.InsertFirstChild(doc.NewElement("annotation"));

            XMLElement* annotation = doc.FirstChildElement("annotation");

            annotation->InsertNewChildElement("folder")->SetText("dataset");
            annotation->InsertNewChildElement("filename")->SetText(p.filename().string().c_str());
            annotation->InsertNewChildElement("path")->SetText(filesystem::absolute(p).string().c_str());
            XMLElement* source = annotation->InsertNewChildElement("source");
            source->InsertNewChildElement("database")->SetText("Unknown");
            XMLElement* size = annotation->InsertNewChildElement("size");

            

            size->InsertNewChildElement("width")->SetText(to_string(frame.cols).c_str());
            size->InsertNewChildElement("height")->SetText(to_string(frame.rows).c_str());
            size->InsertNewChildElement("depth")->SetText(to_string(frame.channels()).c_str());
            annotation->InsertNewChildElement("segmented")->SetText("0");

            for (size_t i = 0; i < indices.size(); ++i)
            {
                int idx = indices[i];
                Rect box = boxes[idx];
                String className = classNamesVec[classIds[idx]];
                putText(frame, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, 8);
                rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);

                XMLElement* object = annotation->InsertNewChildElement("object");
                object->InsertNewChildElement("name")->SetText(className.c_str());
                object->InsertNewChildElement("pose")->SetText("Unspecified");
                object->InsertNewChildElement("truncated")->SetText("0");
                object->InsertNewChildElement("difficult")->SetText("0");

                XMLElement* bndbox = object->InsertNewChildElement("bndbox");
                bndbox->InsertNewChildElement("xmin")->SetText(to_string(box.x).c_str());
                bndbox->InsertNewChildElement("ymin")->SetText(to_string(box.y).c_str());
                bndbox->InsertNewChildElement("xmax")->SetText(to_string(box.x + box.width).c_str());
                bndbox->InsertNewChildElement("ymax")->SetText(to_string(box.y + box.height).c_str());

            }
            string saveName = ".\\test\\" + p.stem().string() + ".xml";
            doc.SaveFile(saveName.c_str());
        }

        imshow("YOLOv4-Detections", frame);
        waitKey(1);
    }
    catch (cv::Exception e) {
        cerr << e.what();
    }
}

int main()
{
    Model yolo;
    yolo.setImgPath("./test");
    yolo.initModel();
    auto begin = filesystem::recursive_directory_iterator(yolo.p);
    auto end = filesystem::recursive_directory_iterator();
    for (auto it = begin; it != end; ++it)
    {
        const string spacer(it.depth() * 2, ' ');   //这个是用来排版的空格
        auto& entry = *it;
        if (filesystem::is_regular_file(entry))     //如果是文件
        {
            string imgPath = filesystem::absolute(entry.path()).string();
            string format(imgPath.end() - 3, imgPath.end());

            if (format == "xml")
                continue;
#ifdef _DEBUG_
            cout << spacer << "File:" << imgPath << endl << format << endl;
            cout << "(" << filesystem::file_size(entry) << " bytes )" << endl;
#endif // _DEBUG_
            yolo.detectObj(entry.path());
        }
        else if (filesystem::is_directory(entry))
        {
#ifdef _DEBUG_
            cout << spacer << "Dir:" << entry << endl;
#endif // _DEBUG_
        }
    }
    return 0;
}
