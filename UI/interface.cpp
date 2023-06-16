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
#include "header1.h"
#include <network.h>

std::string ConvertWideCharToUTF8(const wchar_t* wideCharString)
{
    int utf8Size = WideCharToMultiByte(CP_UTF8, 0, wideCharString, -1, NULL, 0, NULL, NULL);
    std::vector<char> utf8Buffer(utf8Size);
    WideCharToMultiByte(CP_UTF8, 0, wideCharString, -1, utf8Buffer.data(), utf8Size, NULL, NULL);
    return std::string(utf8Buffer.data());
}

// Функция загрузки файла изображения
std::string loadImage(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Получаем размер файла
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        // Создаем буфер для хранения данных файла
        std::string imageData;
        imageData.resize(fileSize);

        // Читаем данные файла в буфер
        file.read(&imageData[0], fileSize);

        // Закрываем файл
        file.close();

        return imageData;
    }
    else {
        throw std::runtime_error("Failed to open file: " + filename);
    }
}

// Функция сохранения файла изображения
void saveImage(const std::string& filename, const std::string& imageData) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Сохраняем данные изображения в файл
        file.write(imageData.data(), imageData.size());

        // Закрываем файл
        file.close();
    }
    else {
        throw std::runtime_error("Failed to save file: " + filename);
    }
}

// Функция для конфигурации размера окна
void configureWindow(int width, int height, const std::string& windowName, int flags = cv::WINDOW_NORMAL) {
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::rectangle(image, cv::Rect(0, 0, width, height), cv::Scalar(255, 255, 255), cv::FILLED);

    cv::namedWindow(windowName, flags);
    cv::imshow(windowName, image);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}

// Функция отрисовки окна (пример)
void drawWindow(int width, int height) {
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::putText(image, "Drawing window", cv::Point(10, height / 2),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    cv::namedWindow("Window", cv::WINDOW_NORMAL);
    cv::imshow("Window", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// Функция для вывода сообщения
void displayMessage(const std::string& message) {
    cv::Mat image = cv::Mat::zeros(200, 600, CV_8UC3);
    cv::putText(image, message, cv::Point(10, 100),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    cv::namedWindow("Message", cv::WINDOW_NORMAL);
    cv::imshow("Message", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// Функция для загрузки окна
void loadWindow(const std::string& windowName) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// Функция для выгрузки окна
void unloadWindow(const std::string& windowName) {
    cv::destroyWindow(windowName);
}

// Функция для конфигурации окна с использованием имени окна
void configureWindowByName(const std::string& windowName, int width, int height, int flags = cv::WINDOW_NORMAL) {
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::rectangle(image, cv::Rect(0, 0, width, height), cv::Scalar(255, 255, 255), cv::FILLED);

    cv::namedWindow(windowName, flags);
    cv::imshow(windowName, image);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}

// Функция для проверки, является ли файл изображением
bool isImageFile(const std::string& filename) {
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    return (extension == "jpg" || extension == "jpeg" || extension == "png" || extension == "bmp");
}


// Функция выбора папки в диалоговом окне
std::string selectFolder(int argc, char *argv[]) {
    QApplication app(argc, argv);  // Создаем экземпляр QApplication

    QFileDialog dialog;  // Создаем диалоговое окно
    dialog.setFileMode(QFileDialog::Directory);  // Устанавливаем режим выбора папки
    dialog.setOption(QFileDialog::ShowDirsOnly);  // Показывать только папки
    // Показываем диалоговое окно и ждем, пока пользователь выберет папку
    if (dialog.exec()) {
        // Получаем выбранную папку
        QStringList selectedFolders = dialog.selectedFiles();
        QString folderPath = selectedFolders.at(0);

        // Преобразуем путь в строку типа std::string и возвращаем
        return folderPath.toStdString();
    }

    // Если пользователь не выбрал папку, возвращаем пустую строку
    return "";
}

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
            onKeyPressed(key, colorGenerator);
        }
    }
    canvas = resizeCanvas(canvas);
    NN recognizer(3, { 500, 20, 10 }, 28 * 28);
    matrix m;
    m.setFromCvMat(canvas);
    recognizer.load("networks.txt");
    int ans = recognizer.recognize(m);
    cv::destroyAllWindows();
    std::cout << "You drew: " << ans << std::endl;
    return 0;
}