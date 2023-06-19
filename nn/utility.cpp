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
    void set_size(int x, int y, int value=0) {
        data.resize(x);
        for(int i = 0; i < x; ++i)
        {
            data[i].resize(y, value);
        }
        size_x = x;
        size_y = y;
    }

    bool isSquare() const {
        return size_x == size_y;
    }
    bool isZeroMatrix() const {
        for(int i = 0; i < size_x; ++i)
            for(int j = 0; j < size_y; ++j)
                if (data[i][j] == 0){
                    return true;
                }
        return false;
    }


    bool operator ==(matrix &other) const {
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

    bool operator !=(matrix &other) const {
        if(size_x != other.get_size_x() || size_y != other.get_size_y()) {
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


    matrix operator *(matrix &other) const
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

    matrix scalar(float n) const{
        std::vector<std::vector<float>> ans(size_x, std::vector<float>(size_y));
        for (int i = 0; i < size_x; ++i)
            for (int j = 0; j < size_y; ++j)
                ans[i][j] = data[i][j] * n;
        return matrix(ans) ;
    }

    matrix activate_relu() const{
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
    matrix gradients() const {
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

    matrix transposeMatrix() const{
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