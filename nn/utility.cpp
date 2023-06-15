#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include "../GPU_stuff/matrix_mul.cpp"
#include "../GPU_stuff/detect_device.cpp"

/**Класс матрицы.
* size_x - размер матрицы по x
* size_y - размер матрицы по y
* data - данные, лежащие в матрице.
*/

class matrix
{

    public:
        matrix(std::vector<std::vector<float>> data_) {
            data = data_;
            size_x = data.size();
            size_y = data[0].size();
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
        * @return data иформация по матрице
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
        */
        void set_size(int x, int y) {
            data.resize(x, std::vector<float>(y));
        }

        /** прогон матрицы через функцию активации relu
        */
        void relu(){
            for (int i = 0; i < size_x; ++i){
                for (int j = 0; j < size_y; ++j){
                    if (data[i][j] < 0){
                        data[i][j] = 0;
                    }
                }
            }
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
        

        matrix operator *(matrix& other) const {
            if(is_cuda()) {
                return matrix(multiplyMatrices(data, other.get_data()));
            }
            else {
                std::vector<std::vector<float>> ans; 
                for(int i = 0; i < size_x; ++i) {
                    for(int j = 0; j < size_y; ++j) {
                        for(int k = 0; k < size_y; ++k) {
                            ans[i][k] += data[i][j] * other.get(j, k);
                        }
                    }
                }
            }
        }


    private:
        int size_x, size_y;
        std::vector<std::vector<float>> data;
};

/**Подсчитывает градиент.
* @param values выходные значения нейронов.
* @return градиент.
*/
std::vector<int> calculate_gradient(std::vector<int> values){
    std::vector<int> gradients(values.size());
    for (int i = 0; i < values.size(); ++i){
        if(values[i] > 0) 
        {
            gradients[i] = 1;
        }
        else 
        {
            gradients[i] = 0;
        }
    }
    return gradients;
}

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



/** конвертирует слои convolution в слои perceptron
*  @param conv_out - то, что вышло из слоев свертки.
*  @return matrix(ans) - матрица, которую нужно подать в перцептрон
*/
matrix conv_to_perc(std::vector<matrix> conv_out) {
    std::vector<std::vector<float>> ans(0);

    for(int i = 0; i < conv_out.size(); ++i) {
        for(int j = 0; j < conv_out[i].get_size_x(); ++j) {
            for(int k = 0; k < conv_out[i].get_size_y(); ++k) {
                ans[0].push_back(conv_out[i].get(j, k));
            }
        }
    }
    return matrix(ans);
}



