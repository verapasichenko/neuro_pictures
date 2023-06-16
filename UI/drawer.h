//
// Created by Идар on 27.05.2023.
//
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stack>
#include <vector>

bool drawing = false;
cv::Point prevPoint;
cv::Mat canvas(512, 512, CV_8UC3, cv::Scalar(255, 255, 255));
std::stack<cv::Mat> undoStack;
std::stack<cv::Mat> redoStack;
bool ctrlPressed = false;
cv::Scalar currentColor(0, 0, 0);
int line_thickness = 3;

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

void onMouse(int event, int x, int y, int, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        drawing = true;
        prevPoint = cv::Point(x, y);
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        undoStack.push(canvas.clone());
        redoStack = std::stack<cv::Mat>();
        drawing = false;
    }
    else if (event == cv::EVENT_MOUSEMOVE)
    {
        if (drawing)
        {
            cv::Point currentPoint(x, y);
            cv::line(canvas, prevPoint, currentPoint, currentColor, line_thickness);
            prevPoint = currentPoint;
            cv::imshow("Editor", canvas);
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
    else if (key == '+' || key == '=')
        line_thickness++;
    else if (key == '-' || key == '_')
        line_thickness = std::max(1, line_thickness - 1);
}

void onCtrlPressed(bool pressed)
{
    ctrlPressed = pressed;
}
