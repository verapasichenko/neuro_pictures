#include "utility.cpp"
#include <vector>
#include <random>
#pragma once


/**Подсчет градиента.
* @param x - точка
* @return значение производной в точке
*/
float calculate_grad(float x) 
{
    return x * (1 - x);
}

/**Подсчет функции активации.
* @param x - точка.
* @return значение функции активации в точке.
*/
float activation_funct(float x) 
{
    return (1.0 / (1.0 + exp(-x)));
}

/**
* Класс слоя перцептрона. Реализован с помощью матричного умножения.
*/
class perceptronLayer 
{
    public:    
        perceptronLayer() 
        {
            weights = matrix(std::vector<std::vector<float>>(0));

            ans = matrix(std::vector<std::vector<float>>(0));
            in = 0;
            out = 0;
        }
        /**Заполнение случайными значениями перцептронного слоя
        * @param in_ - размер входа в слой
        * @param out_ - размер выхода из слоя
        */
        void set(int in_, int out_) 
        {
            in = in_;
            out = out_;
            weights.set_size(in, out);
            for(int i = 0; i < in; ++i) {
                for(int j = 0; j < out; ++j) {
                    weights.set(i, j, (float(rand())/RAND_MAX));
                }
            }
        }
        /**Заполняет слой известными значениями
        * @param values - значения
        */
        void set_values(matrix values) 
        {
            in = values.get_size_x();
            out = values.get_size_y();
            weights = values;
        }
        /**Производит вычисления для слоя нейронов 
        * @param data - значения на входе
        * return значения на выходе из слоя
        */
        matrix predict(matrix data) 
        {
            ans = data * weights;
            for(int j = 0; j < out; ++j) {
                ans.set(0, j, activation_funct(ans.get(0, j)));
            }
            return ans;
        }
        /**Производит вычисление скрытой ошибки.
        * @param error - ошибки на следующем слое
        * @param pweights - веса предыдущего слоя
        * @return ошибка для передачи в предыдущий слой
        */
        matrix calculate_hidden_error(matrix error, matrix pweights)
        {
            matrix hidden_error;
            hidden_error.set_size(1, in);
            for (int i = 0; i < in; ++i) 
            {   
                float err = 0.0;
                for (int j = 0; j < out; ++j) 
                {
                    err += error.get(0, j) * pweights.get(i, j) * calculate_grad(ans.get(0, j));
                }
                hidden_error.set(0, i, err);
            }

            return hidden_error;
        }
        
        /**Производит обновление весов слоя с учетом ошибки
        * @param error - ошибки со следующего слоя
        * @param learning_rate - скорость обучения
        */
        void update_weights(matrix error, float learning_rate)
        {
            for (int i = 0; i < in; ++i)
            {
                for (int j = 0; j < out; ++j)
                {
                    weights.add(i, j, learning_rate * error.get(0, i) * calculate_grad(ans.get(0, j)));
                }
            }
        }
        matrix get_weights() {
            return weights;
        }
        int get_size_x() 
        {
            return weights.get_size_x();
        }
        int get_size_y() 
        {
            return weights.get_size_y();
        }

    private:
        int in;
        int out;
        matrix weights;
        matrix ans;
};
