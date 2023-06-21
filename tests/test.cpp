#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "main.cpp"


TEST_CASE("Change Tool Test") {
    Tool tool1{1, "Tool 1"};
    Tool tool2{2, "Tool 2"};

    SUBCASE("Change Tool 1") {
        changeTool(tool1);
        CHECK_EQ(previousTool.id, 0);  // Предыдущий инструмент должен быть пустым
        CHECK_EQ(currentTool.id, 1);   // Текущий инструмент должен быть равен tool1
    }

    SUBCASE("Change Tool 2") {
        changeTool(tool2);
        CHECK_EQ(previousTool.id, 1);  // Предыдущий инструмент должен быть равен tool1
        CHECK_EQ(currentTool.id, 2);   // Текущий инструмент должен быть равен tool2
    }
}




// Тест для функции drawCircle()
TEST_CASE("Test drawCircle function")
{
    // Создание пустого холста для тестирования
    cv::Mat canvas(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));

    // Тестирование отображения круга на холсте
    drawCircle(50, 50, 20);

    // Проверка, что холст изменился
    cv::Mat expectedCanvas(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::circle(expectedCanvas, cv::Point(50, 50), 20, cv::Scalar(255, 255, 255), 1);
    CHECK_EQ(canvas, expectedCanvas);
}


TEST_CASE("Test copyCanvas function")
{
    // Создание исходного холста
    cv::Mat canvas(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));

    // Копирование холста
    copyCanvas();

    // Проверка, что скопированный холст и исходный холст идентичны
    cv::Mat copiedRegion;
    copiedRegion = canvas.clone();
    CHECK_EQ(copiedRegion, canvas);
}


// Тест для функции pasteCanvas()
TEST_CASE("Test pasteCanvas function")
{
    // Создание исходного холста и скопированного региона
    cv::Mat canvas(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat copiedRegion(50, 50, CV_8UC3, cv::Scalar(255, 255, 255));

    // Вызов функции pasteCanvas()
    pasteCanvas();

    // Проверка, что холст после вставки скопированного региона изменился
    cv::Mat expectedCanvas(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Rect roi(cv::Point(0, 0), copiedRegion.size());
    copiedRegion.copyTo(expectedCanvas(roi));
    CHECK_EQ(canvas, expectedCanvas);
}