﻿#include <opencv2/opencv.hpp>
#include <stack>
#include <vector>
#include <cmath>


bool drawing = false;
cv::Point prevPoint;
cv::Mat canvas(400, 400, CV_8UC4, cv::Scalar(255, 255, 255, 255));
std::stack<cv::Mat> undoStack;
std::stack<cv::Mat> redoStack;
bool ctrlPressed = false;
cv::Scalar currentColor(0, 0, 0, 255);
int lineThickness = 3;
int pencilAlpha = 255;


class ColorGenerator
{
private:
    std::vector<cv::Scalar> colors;
    size_t currentIndex;

public:
    ColorGenerator() : currentIndex(0)
    {
        colors = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 0) };
    }
    cv::Scalar getNextColor()
    {
        const cv::Scalar& color = colors[currentIndex];
        currentIndex = (currentIndex + 1) % colors.size();
        return color;
    }
};

enum class Tool
{
    Pencil,
    Line,
    Eraser,
    WaveLine,
    Circle,
    Triangle,
    Rectangle,
};

Tool currentTool = Tool::Pencil;
Tool previousTool = Tool::Pencil;
cv::Mat copiedRegion;
cv::Point startPoint;

void changeLineThickness(int thickness)
{
    lineThickness = std::max(1, thickness);
}

void copyCanvas()
{
    copiedRegion = canvas.clone();
}

void pasteCanvas()
{
    if (!copiedRegion.empty())
    {
        undoStack.push(canvas.clone());
        redoStack = std::stack<cv::Mat>();
        cv::Rect roi(cv::Point(0, 0), copiedRegion.size());
        copiedRegion.copyTo(canvas(roi));
        cv::imshow("Editor", canvas);
    }
}

void flipCanvas()
{
    cv::flip(canvas, canvas, 1);
    cv::imshow("Editor", canvas);
}

void drawRectangle(int x, int y, int width, int height)
{
    cv::rectangle(canvas, cv::Point(x, y), cv::Point(x + width, y + height), currentColor, lineThickness);
    cv::imshow("Editor", canvas);
}

void drawWaveLine(int x1, int y1, int x2, int y2)
{
    const float waveAmplitude = 10.0f;
    const float waveFrequency = 0.1f;

    float length = cv::norm(cv::Point(x2 - x1, y2 - y1));
    int numSegments = static_cast<int>(length);

    cv::Point prevPoint(x1, y1);

    for (int i = 1; i <= numSegments; ++i)
    {
        float t = static_cast<float>(i) / numSegments;
        float displacement = waveAmplitude * std::sin(2 * CV_PI * waveFrequency * t);

        int x = static_cast<int>(x1 + t * (x2 - x1));
        int y = static_cast<int>(y1 + t * (y2 - y1) + displacement);

        cv::Scalar waveColor = cv::Scalar(currentColor[0], currentColor[1], currentColor[2], pencilAlpha);
        cv::line(canvas, prevPoint, cv::Point(x, y), waveColor, lineThickness);
        prevPoint = cv::Point(x, y);
    }

    cv::imshow("Editor", canvas);
}

void drawCircle(int x, int y, int radius)
{
    cv::Scalar circleColor = cv::Scalar(currentColor[0], currentColor[1], currentColor[2], pencilAlpha);
    cv::circle(canvas, cv::Point(x, y), radius, circleColor, lineThickness);
    cv::imshow("Editor", canvas);
}

void drawTriangle(int x, int y, int sideLength)
{
    const float sqrt3 = std::sqrt(3);

    cv::Point p1(x, y - sideLength / (2 * sqrt3));
    cv::Point p2(x - sideLength / 2, y + sideLength / (2 * sqrt3));
    cv::Point p3(x + sideLength / 2, y + sideLength / (2 * sqrt3));

    cv::line(canvas, p1, p2, currentColor, lineThickness);
    cv::line(canvas, p2, p3, currentColor, lineThickness);
    cv::line(canvas, p3, p1, currentColor, lineThickness);

    cv::imshow("Editor", canvas);
}

void onMouse(int event, int x, int y, int, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (currentTool == Tool::Pencil || currentTool == Tool::Eraser)
        {
            drawing = true;
            prevPoint = cv::Point(x, y);
        }
        else if (currentTool == Tool::Line)
        {
            startPoint = cv::Point(x, y);
        }
        else if (currentTool == Tool::WaveLine)
        {
            drawing = true;
            prevPoint = cv::Point(x, y);
        }
        else if (currentTool == Tool::Rectangle)
        {
            startPoint = cv::Point(x, y);
        }
        else if (currentTool == Tool::Circle)
        {
            startPoint = cv::Point(x, y);
        }
        else if (currentTool == Tool::Triangle)
        {
            startPoint = cv::Point(x, y);
        }
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        if (currentTool == Tool::Pencil || currentTool == Tool::Eraser)
        {
            undoStack.push(canvas.clone());
            redoStack = std::stack<cv::Mat>();
            drawing = false;
        }
        else if (currentTool == Tool::Line)
        {
            cv::line(canvas, startPoint, cv::Point(x, y), currentColor, lineThickness);
            cv::imshow("Editor", canvas);
        }
        else if (currentTool == Tool::WaveLine)
        {
            drawing = false;
            drawWaveLine(startPoint.x, startPoint.y, x, y);
        }
        else if (currentTool == Tool::Rectangle)
        {
            int width = x - startPoint.x;
            int height = y - startPoint.y;
            drawRectangle(startPoint.x, startPoint.y, width, height);
        }
        else if (currentTool == Tool::Circle)
        {
            int radius = cv::norm(startPoint - cv::Point(x, y));
            drawCircle(startPoint.x, startPoint.y, radius);
        }
        else if (currentTool == Tool::Triangle)
        {
            int sideLength = cv::norm(startPoint - cv::Point(x, y));
            drawTriangle(startPoint.x, startPoint.y, sideLength);
        }
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        if (drawing)
        {
            cv::Point currentPoint(x, y);
            if (currentTool == Tool::Pencil)
            {
                cv::Scalar pencilColor = cv::Scalar(currentColor[0], currentColor[1], currentColor[2], pencilAlpha);
                cv::line(canvas, prevPoint, currentPoint, pencilColor, lineThickness);
                prevPoint = currentPoint;
                cv::imshow("Editor", canvas);
            }
            else if (currentTool == Tool::Eraser)
            {
                cv::line(canvas, prevPoint, currentPoint, cv::Scalar(255, 255, 255, 0), lineThickness);
                prevPoint = currentPoint;
                cv::imshow("Editor", canvas);
            }
            else if (currentTool == Tool::WaveLine)
            {
                drawWaveLine(prevPoint.x, prevPoint.y, currentPoint.x, currentPoint.y);
                prevPoint = currentPoint;
            }
        }
    }
}


void undo()
{
    if (!undoStack.empty())
    {
        redoStack.push(canvas.clone());
        canvas = undoStack.top().clone();
        undoStack.pop();
        cv::imshow("Editor", canvas);
    }
}

void redo()
{
    if (!redoStack.empty())
    {
        undoStack.push(canvas.clone());
        canvas = redoStack.top().clone();
        redoStack.pop();
        cv::imshow("Editor", canvas);
    }
}

void changeColor(ColorGenerator& colorGenerator)
{
    currentColor = colorGenerator.getNextColor();
}

void changeTool(Tool tool)
{
    previousTool = currentTool;
    currentTool = tool;
}

void onKeyPressed(int key, ColorGenerator& colorGenerator)
{
    if (key == 'c') // Clear canvas
    {
        undoStack.push(canvas.clone());
        redoStack = std::stack<cv::Mat>();
        canvas = cv::Scalar(255, 255, 255);
        cv::imshow("Editor", canvas);
    }
    else if (key == 'z' && ctrlPressed)
    {
        undo();
    }
    else if (key == 'y' && ctrlPressed)
    {
        redo();
    }
    else if (key == 'r')
    {
        changeColor(colorGenerator);
    }
    else if (key == '+')
    {
        changeLineThickness(lineThickness + 1);
    }
    else if (key == '-')
    {
        changeLineThickness(std::max(1, lineThickness - 1));
    }
    else if (key == 'x')
    {
        copyCanvas();
    }
    else if (key == 'v')
    {
        pasteCanvas();
    }
    else if (key == 'f')
    {
        flipCanvas();
    }
    else if (key == 'g')
    {
        changeTool(Tool::Line);
    }
    else if (key == 'e')
    {
        if (currentTool == Tool::Eraser)
        {
            changeTool(previousTool);
        }
        else
        {
            changeTool(Tool::Eraser);
        }
    }
    else if (key == 'p')
    {
        changeTool(Tool::Pencil);
    }
    else if (key == 'w')
    {
        changeTool(Tool::WaveLine);
    }
    else if (key == 'o')
    {

        drawCircle(200, 200, 50); // Пример координат и радиуса;
    }
    else if (key == 't')
    {

        drawTriangle(100, 100, 50); // Пример координат и длины сторон;

    }

    else if (key == 'i')
    {
        static bool showWindow = false;
        showWindow = !showWindow;
        if (showWindow)
        {
            cv::namedWindow("Info");
            cv::moveWindow("Info", 500, 200);
            cv::putText(canvas, "привет!" , cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
            cv::imshow("Editor", canvas);
        }
        else
        {
            cv::destroyWindow("Info");
        }
    }
}

void onCtrlPressed(bool pressed)
{
    ctrlPressed = pressed;
}
void drawTriangle(int x, int y, int sideLength);

