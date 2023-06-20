//
// Created by Иван Грибанов on 11.06.2023.
//

#include <iostream>
#include <windows.h>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFileDialog>
#include <QtCore/QString>
#include "dirent.h"
#include "drawer.h"
#include <network.h>

void displayText(const std::string& text, const cv::Scalar& textColor, const cv::Scalar& bgColor,
                 const std::string& fontName, int fontSize, bool bold, bool italic, bool underline,
                 bool strikethrough, bool shadow) {
    cv::Mat image(200, 400, CV_8UC3, bgColor);

    int baseline = 0;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    if (fontName == "Arial") {
        fontFace = cv::FONT_HERSHEY_PLAIN;
    }
    else if (fontName == "Times New Roman") {
        fontFace = cv::FONT_HERSHEY_SIMPLEX;
    }

    int fontStyle = cv::FONT_HERSHEY_SIMPLEX;
    if (bold) {
        fontStyle |= cv::FONT_HERSHEY_SIMPLEX;
    }
    if (italic) {
        fontStyle |= cv::FONT_HERSHEY_SIMPLEX;
    }
    if (underline) {
        cv::putText(image, text, cv::Point(10, 200), fontFace, fontSize, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    }
    if (strikethrough) {
        cv::putText(image, text, cv::Point(10, 250), fontFace, fontSize, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    }

    if (shadow) {
        cv::putText(image, text, cv::Point(11, 200), fontFace, fontSize, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    }

    cv::Size textSize = cv::getTextSize(text, fontFace, fontSize, 2, &baseline);

    cv::Point textOrg((image.cols - textSize.width) / 2, (image.rows + textSize.height) / 2);

    cv::putText(image, text, textOrg, fontFace, fontSize, textColor, 2, fontStyle);

    cv::namedWindow("Text Display", cv::WINDOW_NORMAL);
    cv::imshow("Text Display", image);

    cv::waitKey(0);
    cv::destroyWindow("Text Display");
}


/**
 * Измените размер изображения на холсте до указанного размера.
 *
 * Функция resizeCanvas() изменяет размер входного изображения canvas до указанного размера с помощью билинейной интерполяции.
 * Измененное изображение canvas возвращается как новый объект cv::Mat.
 *
 * @param canvas Входное изображение canvas, размер которого необходимо изменить.
 * @return Изображение на холсте с измененным размером.
 */
cv::Mat resizeCanvas(const cv::Mat& canvas)
{
    cv::Mat resizedCanvas;
    cv::resize(canvas, resizedCanvas, cv::Size(28, 28));
    return resizedCanvas;
}


int main(int argc, char* argv[]) {
    cv::namedWindow("Editor");
    cv::setMouseCallback("Editor", onMouse);

    ColorGenerator colorGenerator;
    BackgroundGenerator backgroundGenerator;


    while (true)
    {
        cv::imshow("Editor", canvas);
        int key = cv::waitKey(1);

        if (key == 27)
            break;
        else if (key == 224)
        {
            onCtrlPressed(true);
        }
        else if (key == -224)
        {
            onCtrlPressed(false);
        }
        else
        {
            onKeyPressed(key, colorGenerator, backgroundGenerator);
        }
    }
    canvas = resizeCanvas(canvas);
    NN recognizer(3, { 500, 20, 10 }, 28 * 28);
    matrix m;
    m.setFromCvMat(canvas);
    std::string filename = "D:/projects/interface/network.txt";
    recognizer.load(filename);
    int ans = recognizer.recognize(m);
    cv::destroyAllWindows();
    std::cout << "You drew: " << ans << std::endl;

    std::string text = "You drew " + std::to_string(ans);
    cv::Scalar textColor(0, 0, 0);
    cv::Scalar bgColor(255, 255, 255);

    std::string fontName = "Arial";
    int fontSize = 3;
    bool bold = false;
    bool italic = false;
    bool underline = false;
    bool strikethrough = false;
    bool shadow = false;

    displayText(text, textColor, bgColor, fontName, fontSize, bold, italic, underline, strikethrough, shadow);
    return 0;
}