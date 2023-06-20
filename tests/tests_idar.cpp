//
// Created by Идар on 20.06.2023.
//
#include "../UI/drawer.h"
#include "doctest.h"

TEST_CASE("ColorGenerator getNextColor")
{
    ColorGenerator colorGenerator;
    cv::Scalar color1 = colorGenerator.getNextColor();
    cv::Scalar color2 = colorGenerator.getNextColor();
    cv::Scalar color3 = colorGenerator.getNextColor();
    cv::Scalar expectedColor1(255, 0, 0);
    cv::Scalar expectedColor2(0, 255, 0);
    cv::Scalar expectedColor3(0, 0, 255);
    CHECK(color1 == expectedColor1);
    CHECK(color2 == expectedColor2);
    CHECK(color3 == expectedColor3);
    cv::Scalar color4 = colorGenerator.getNextColor();
    cv::Scalar expectedColor4(0, 0, 0);
    CHECK(color4 == expectedColor4);
    cv::Scalar color5 = colorGenerator.getNextColor();
    cv::Scalar color6 = colorGenerator.getNextColor();
    CHECK(color5 == expectedColor1);
    CHECK(color6 == expectedColor2);
}

TEST_CASE("BackgroundGenerator getNextBackground")
{
    BackgroundGenerator bgGenerator;
    cv::Scalar background1 = bgGenerator.getNextBackground();
    CHECK(background1 == cv::Scalar(255, 255, 255));
    cv::Scalar background2 = bgGenerator.getNextBackground();
    CHECK(background2 == cv::Scalar(0, 0, 0));
    cv::Scalar background3 = bgGenerator.getNextBackground();
    CHECK(background3 == cv::Scalar(128, 128, 128));
    cv::Scalar background4 = bgGenerator.getNextBackground();
    CHECK(background4 == cv::Scalar(255, 255, 255));
}

TEST_CASE("Нажатая ЛКМ")
{
    onMouse(cv::EVENT_LBUTTONDOWN, 100, 100, 0, nullptr);
    CHECK(drawing == true);
    CHECK(prevPoint == cv::Point(100, 100));
}

TEST_CASE("Не нажатая ЛКМ")
{
    onMouse(cv::EVENT_LBUTTONUP, 200, 200, 0, nullptr);
    CHECK(drawing == false);
    CHECK(undoStack.size() == 1);
    CHECK(redoStack.empty() == true);
}

TEST_CASE("Движение мыши")
{
    drawing = true;
    prevPoint = cv::Point(150, 150);
    canvas = cv::Mat(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::line(canvas, cv::Point(150, 150), cv::Point(200, 200), currentColor, line_thickness);
    onMouse(cv::EVENT_MOUSEMOVE, 200, 200, 0, nullptr);
    CHECK(canvas.at<cv::Vec3b>(200, 200) == cv::Vec3b(0, 0, 0));
    CHECK(prevPoint == cv::Point(200, 200));
}

TEST_CASE("Чистка холста")
{
    ColorGenerator colorGenerator;
    BackgroundGenerator backgroundGenerator;
    undoStack.push(cv::Mat(400, 400, CV_8UC3, cv::Scalar(0, 0, 0)));
    redoStack.push(cv::Mat(400, 400, CV_8UC3, cv::Scalar(255, 255, 255)));
    onKeyPressed('c', colorGenerator, backgroundGenerator);
    CHECK(canvas.empty() == false);
    CHECK(cv::countNonZero(canvas == cv::Scalar(255, 255, 255)) == 400 * 400 * 3);
    CHECK(undoStack.size() == 2);
    CHECK(redoStack.empty() == true);
}
