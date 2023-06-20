#include <vector>
#pragma once

/**Класс матрицы.
* size_x - размер матрицы по x
* size_y - размер матрицы по y
* data - данные, лежащие в матрице.
*/
class matrix
{
public:
    matrix()
    {
        data = std::vector<std::vector<float>>(0);
        size_x = 0;
        size_y = 0;
    }

    matrix(std::vector<std::vector<float>> data_)
    {
        data = data_;
        size_x = data.size();
        if(size_x == 0)
        {
            size_y = 0;
        }
        else
        {
            size_y = data[0].size();
        }
    }

    /**геттер размера
    * @return size_x размер матрицы по x
    */
    int get_size_x() {
        return size_x;
    }
    /**геттер размера
    * @return size_y размер матрицы по y
    */
    int get_size_y() {
        return size_y;
    }
    /**геттер данных
    * @return data информация по матрице
    */
    std::vector<std::vector<float>> get_data() {
        return data;
    }
    /**геттер числа из матрицы
    * @return data[x][y] - число на нужной позиции
    */
    float get(int x, int y) {
        return data[x][y];
    }

    /** сеттер значения.
    *  @param x - координата x значения
    *  @param y - координата y значения
    *  @param val - значение, на которое заменить
    */
    void set(int x, int y, float val) {
        data[x][y] = val;
    }
    /** += к числу
    *  @param x - координата x значения
    *  @param y - координата y значения
    *  @param val - значение, которое добавить
    */
    void add(int x, int y, float val) {
        data[x][y] += val;
    }
    /** сеттер размера.
    *  @param x - размер по x
    *  @param y - размер по y
    *  ВНИМАНИЕ: Удаляет исходную матрицу
    */
    void set_size(int x, int y, int value = 0) {
        data.resize(x);
        for(int i = 0; i < x; ++i)
        {
            data[i].resize(y, value);
        }
        size_x = x;
        size_y = y;
    }
    /**Проверка на квадратность матрицы.
     * @return равенство по длине и ширине.
     */
    bool isSquare() const {
        return size_x == size_y;
    }
    /**Проверка на нулевность матрицы.
     * @return true, если матрица состоит из нулей.
     * @return false, если матрица не состоит из нулей.
     */
    bool isZeroMatrix() const {
        for(int i = 0; i < size_x; ++i)
            for(int j = 0; j < size_y; ++j)
                if (data[i][j] == 0){
                    return true;
                }
        return false;
    }
    /**Перегруженный оператор "равно"
     * @param matrix& - ссылка на объект matrix
     * @param other - другая матрица
     * @return true, если матрицы равны.
     * @return false, если матрицы не равны.
     */
    bool operator ==(matrix& other) const {
        if(size_x != other.get_size_x() || size_y != other.get_size_y()) {
            return false;
        }
        for(int i = 0; i < size_x; ++i) {
            for(int j = 0; j < size_y; ++j) {
                if(data[i][j] != other.get(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
    /**Перегруженный оператор "не равно"
    * @param matrix& - ссылка на объект matrix
    * @param other - другая матрица
    * @return true, если матрицы не равны.
    * @return false, если матрицы равны.
    */
    bool operator !=(matrix& other) const { //ссылка
        if(size_x != other.get_size_x() || size_y != other.get_size_y()) { //или
            return true;
        }
        for(int i = 0; i < size_x; ++i) {
            for(int j = 0; j < size_y; ++j) {
                if(data[i][j] != other.get(i, j)) {
                    return true;
                }
            }
        }
        return false;
    }
    /**Перегруженный оператор "умножение"
    * @param matrix& - ссылка на объект matrix
    * @param other - другая матрица
    * @return ans - матрица, полученная путем перемножения двух матриц.
    */
    matrix operator *(matrix& other) const
    {
        std::vector<std::vector<float>> ans(size_x, std::vector<float>(other.get_size_y()));
        for(int i = 0; i < size_x; ++i) {
            for(int j = 0; j < size_y; ++j) {
                for(int k = 0; k < other.get_size_y(); ++k) {
                    ans[i][k] += data[i][j] * other.get(j, k);
                }
            }
        }
        return matrix(ans);
    }
    /**Перегруженный оператор "сложение"
    * @param matrix& - ссылка на объект matrix
    * @param other - другая матрица
    * @return ans - матрица, полученная путем сложения двух матриц.
    */
    matrix operator +(matrix& other) const
    {
        std::vector<std::vector<float>> ans(size_x, std::vector<float>(other.get_size_y()));
        for(int i = 0; i < size_x; ++i) {
            for(int j = 0; j < size_y; ++j) {
                for(int k = 0; k < other.get_size_y(); ++k) {
                    ans[i][k] += data[i][j] + other.get(j, k);
                }
            }
        }
        return matrix(ans);
    }
    /**Умножение матрицы на число.
    * @param n - какое-либо число.
    * @return ans - матрица, полученная путем умножения матрицы на число.
    */
    matrix scalar(float n) {
        std::vector<std::vector<float>> ans(size_x, std::vector<float>(size_y));
        for (int i = 0; i < size_x; ++i)
            for (int j = 0; j < size_y; ++j)
                ans[i][j] = data[i][j] * n;
        return matrix(ans) ;
    }
    /**Активация.
    * @return new_matrix - матрица, полученная путем замены некоторых символов.
    */
    matrix activate_relu() {
        std::vector<std::vector<float>> new_matrix(size_x, std::vector<float>(size_y));
        for (int i = 0; i < size_x; ++i)
            for (int j = 0; j < size_y; ++j)
                if (data[i][j] < 0){
                    new_matrix[i][j] = 0;
                }
                else{
                    new_matrix[i][j] = data[i][j];
                }
        return matrix(new_matrix);
    }
    /**Градиент.
    * @return gradient - матрица, полученная путем замены некоторых символов.
    */
    matrix gradients() {
        std::vector<std::vector<float>> gradient(size_x, std::vector<float>(size_y));
        for (int i = 0; i < size_x; ++i)
            for (int j = 0; j < size_y; ++j)
                if (data[i][j] > 0){
                    gradient[i][j] = 1;
                }
                else{
                    gradient[i][j] = 0;
                }
        return matrix(gradient);
    }
    /**Транспонирование матрицы.
    * @return transposedMatrix - транспонированная матрица.
    */
    matrix transposeMatrix() {
        std::vector<std::vector<float>> transposedMatrix(size_y, std::vector<float>(size_x));
        for (int i = 0; i < size_x; ++i)
            for (int j = 0; j < size_y; ++j)
                transposedMatrix[j][i] = data[i][j];
        return matrix(transposedMatrix);
    }

    /**Находит ошибку между значениями векторов.
    * @param output_res Выходное значение нейронной сети.
    * @param result Ожидаемое(истинное) значение.
    * @return Ошибка.
    */
    matrix calculate_error(matrix output_res, matrix result)
    {
        matrix err;
        err.set_size(1, output_res.get_size_y());
        for (int i = 0; i < result.get_size_y(); ++i)
        {
            err.set(0, i, (result.get(0, i) - output_res.get(0, i)) * (result.get(0, i) - output_res.get(0, i)));
        }
        return err;
    }
    /**Вычисление суммы всех элементов матрицы.
    * @return Сумма всех элементов.
    */
    float sum() {
        float sum = 0.0;
        for (int i = 0; i < size_x; ++i)
            for (int j = 0; j < size_y; ++j)
                sum += data[i][j];
        return sum;
        }
    /**Вычисление среднего значения из всех элементов матрицы.
    * @return Среднее значение из всех элементов матрицы.
    */
    float mean() {
        float sum = 0.0;
        int count = 0;
        for (int i = 0; i < size_x; ++i){
            for (int j = 0; j < size_y; ++j){
                sum += data[i][j];
                count++;
            }
        }
        return sum / count;
        }
private:
    int size_x, size_y;
    std::vector<std::vector<float>> data;
};

/**Находит максимальное число.
         * @param a первое число.
         * @param b второе число.
         * @param c третье число.
         * @param d четвертое число.
         * @return максимальное число.
         */
int max(int a, int b, int c, int d)
{
    return std::max(std::max(a, b), std::max(c, d));
}