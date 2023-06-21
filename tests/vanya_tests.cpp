#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <opencv2/opencv.hpp>

void drawTriangle(cv::Mat& canvas, int x, int y, int sideLength)
{
    const float sqrt3 = std::sqrt(3);

    cv::Point p1(x, y - sideLength / (2 * sqrt3));
    cv::Point p2(x - sideLength / 2, y + sideLength / (2 * sqrt3));
    cv::Point p3(x + sideLength / 2, y + sideLength / (2 * sqrt3));

    cv::line(canvas, p1, p2, cv::Scalar(0, 0, 0, 255), 1);
    cv::line(canvas, p2, p3, cv::Scalar(0, 0, 0, 255), 1);
    cv::line(canvas, p3, p1, cv::Scalar(0, 0, 0, 255), 1);

    cv::imshow("Editor", canvas);
}

TEST_CASE("Test Triangle Drawing")
{
    // Создаем холст и устанавливаем начальное состояние
    cv::Mat canvas(400, 400, CV_8UC4, cv::Scalar(255, 255, 255, 255));
    cv::Point startPoint(100, 100);
    int sideLength = 50;
    drawTriangle(canvas, startPoint.x, startPoint.y, sideLength);

    // Проверяем, что треугольник правильно нарисован
    SUBCASE("Check Triangle Pixels")
    {
        // Проверяем пиксели на границе треугольника
        for (int i = 0; i <= sideLength; ++i)
        {
            CHECK(canvas.at<cv::Vec4b>(startPoint.y - sideLength / (2 * std::sqrt(3)) + i, startPoint.x + i) == cv::Vec4b(0, 0, 0, 255));
            CHECK(canvas.at<cv::Vec4b>(startPoint.y + sideLength / (2 * std::sqrt(3)), startPoint.x - sideLength / 2 + i) == cv::Vec4b(0, 0, 0, 255));
            CHECK(canvas.at<cv::Vec4b>(startPoint.y + sideLength / (2 * std::sqrt(3)), startPoint.x + sideLength / 2 - i) == cv::Vec4b(0, 0, 0, 255));
        }

        // Проверяем пиксели внутри треугольника
        for (int i = 1; i <= sideLength; ++i)
        {
            for (int j = 1; j < i; ++j)
            {
                CHECK(canvas.at<cv::Vec4b>(startPoint.y - sideLength / (2 * std::sqrt(3)) + j, startPoint.x + i) == cv::Vec4b(0, 0, 0, 255));
                CHECK(canvas.at<cv::Vec4b>(startPoint.y + sideLength / (2 * std::sqrt(3)) - j, startPoint.x - sideLength / 2 + i) == cv::Vec4b(0, 0, 0, 255));
                CHECK(canvas.at<cv::Vec4b>(startPoint.y + sideLength / (2 * std::sqrt(3)) - j, startPoint.x + sideLength / 2 - i) == cv::Vec4b(0, 0, 0, 255));
            }
        }
    }

    // Проверяем, что другие пиксели на холсте остались неизменными
    SUBCASE("Check Canvas Pixels")
    {
        // Проверяем пиксели вокруг треугольника
        for (int i = 0; i < canvas.rows; ++i)
        {
            for (int j = 0; j < canvas.cols; ++j)
            {
                if ((i >= startPoint.y - sideLength / (2 * std::sqrt(3)) && i <= startPoint.y + sideLength / (2 * std::sqrt(3))) && (j >= startPoint.x - sideLength / 2 && j <= startPoint.x + sideLength / 2))
                {
                    CHECK(canvas.at<cv::Vec4b>(i, j) == cv::Vec4b(255, 255, 255, 255));
                }
            }
        }
    }
}
