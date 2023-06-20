#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stack>
#include <vector>

bool drawing = false;
cv::Point prevPoint;
cv::Mat canvas(400, 400, CV_8UC3, cv::Scalar(255, 255, 255));
std::stack<cv::Mat> undoStack;
std::stack<cv::Mat> redoStack;
bool ctrlPressed = false;
cv::Scalar currentColor(0, 0, 0);
int line_thickness = 3;

/**
 * Генератор цветов.
 * Класс ColorGenerator предоставляет функционал для генерации последовательности цветов.
 * При каждом вызове метода getNextColor() возвращается следующий цвет из заранее заданного
 * набора цветов.
 * Приватные члены класса:
 * - colors: вектор, содержащий заранее заданные цвета.
 * - currentIndex: индекс текущего цвета в векторе colors.
 * Публичные методы класса:
 * - ColorGenerator(): конструктор класса, инициализирует вектор colors и устанавливает
 *   начальное значение индекса currentIndex.
 * - getNextColor(): возвращает следующий цвет из вектора colors и обновляет значение
 *   индекса currentIndex.
 */
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

/**
 * Background generator.
 * The BackgroundGenerator class provides functionality for generating and changing the
 * background color of the canvas.
 * Private class members:
 * - backgrounds: a vector containing the predefined background colors.
 * - currentIndex: the index of the current background color in the backgrounds vector.
 * Public class methods:
 * - BackgroundGenerator(): the class constructor initializes the backgrounds vector and sets
 *   the initial value of the currentIndex.
 * - getNextBackground(): returns the next background color from the backgrounds vector and updates
 *   the currentIndex value.
 */
class BackgroundGenerator
{
private:
    std::vector<cv::Scalar> backgrounds;
    size_t currentIndex;

public:
    BackgroundGenerator() : currentIndex(0)
    {
        backgrounds = { cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 0), cv::Scalar(128, 128, 128) };
    }

    cv::Scalar getNextBackground()
    {
        const cv::Scalar& background = backgrounds[currentIndex];
        currentIndex = (currentIndex + 1) % backgrounds.size();
        return background;
    }
};

/**
 * Обработчик событий мыши.
 *
 * Обрабатывает нажатие и отпускание левой кнопки мыши,
 * а также перемещение мыши.
 *
 * @param event    Тип события мыши.
 * @param x        Координата X точки события.
 * @param y        Координата Y точки события.
 * @param          Дополнительные параметры (не используются).
 * @param userdata Данные пользователя (не используются).
 */
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

/**
 * Отменяет последнее действие на холсте.
 * Если стек отмены (`undoStack`) не пустой, то текущее состояние холста клонируется
 * и добавляется в стек повтора (`redoStack`). Затем верхний элемент из стека отмены
 * извлекается и устанавливается в качестве нового состояния холста. Обновленный холст
 * отображается в окне редактора (`"Editor"`).
 */
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

/**
 * Восстановление последнего отмененного действия.
 * Функция redo() восстанавливает последнее отмененное действие из стека redoStack, если он не пуст.
 * Состояние холста сохраняется в стеке undoStack, затем извлекается последнее сохраненное состояние
 * холста из redoStack, присваивается переменной canvas и удаляется из стека redoStack.
 * Обновленное состояние холста отображается в окне "Editor".
 */
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

/**
 * Изменение текущего цвета рисования.
 *
 * Функция changeColor() изменяет текущий цвет рисования на следующий цвет, полученный
 * из объекта colorGenerator.
 *
 * @param colorGenerator Объект ColorGenerator, используемый для генерации следующего цвета.
 */
void changeColor(ColorGenerator& colorGenerator)
{
    currentColor = colorGenerator.getNextColor();
}

/**
 * Изменение цвета фона холста.  
 *
 * Функция changeBackground() изменяет цвет фона холста на
 * получено из объекта backgroundGenerator.

 *
 * @param backgroundGenerator Объект BackgroundGenerator, используемый для создания следующего цвета фона.
 */
void changeBackground(BackgroundGenerator& backgroundGenerator)
{
    cv::Scalar backgroundColor = backgroundGenerator.getNextBackground();
    canvas.setTo(backgroundColor);
    cv::imshow("Editor", canvas);
}

/**
 * Обработчик нажатия клавиш.
 *
 * Функция `onKeyPressed()` обрабатывает нажатия клавиш и выполняет соответствующие действия.
 * - 'c': Очищает холст. Текущее состояние холста клонируется и добавляется в стек отмены (`undoStack`).
 *        Стек повтора (`redoStack`) очищается.
 * - 'z' + Ctrl: Вызывает функцию `undo()`
 * - 'y' + Ctrl: Вызывает функцию `redo()`
 * - 'r': Вызывает функцию `changeColor()`,
 * - '+' или '=': Увеличивает толщину линии на единицу.
 * - '-' или '_': Уменьшает толщину линии на единицу, но не менее 1.
 * @param key             Код нажатой клавиши.
 * @param colorGenerator  Объект ColorGenerator, используемый для изменения цвета рисования.
 */
void onKeyPressed(int key, ColorGenerator& colorGenerator, BackgroundGenerator& backgroundGenerator)
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
    else if (key == 'b')
    {
        changeBackground(backgroundGenerator);
    }
    else if (key == '+' || key == '=')
        line_thickness++;
    else if (key == '-' || key == '_')
        line_thickness = std::max(1, line_thickness - 1);
}
/**
 * Установка состояния нажатия клавиши Ctrl.
 * @param pressed  Состояние клавиши Ctrl: true (нажата) или false (отпущена).
 */
void onCtrlPressed(bool pressed)
{
    ctrlPressed = pressed;
}