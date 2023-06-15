#include <vector>
#include <fstream>
#include <string>
#include "perceptron.cpp"
#include "utility.cpp"
#pragma once

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


/**Класс нейросети
* .recognise() - определение класса
* .learn() - обучение нейросети
*/
class NN 
{
    public:
        /**Генерирут случайные веса всем слоям.
        */
        NN(int numberOfLayersP_, std::vector<int> p_sizes, int first_in) 
        {
            number_of_layers_prc = numberOfLayersP_;
            prc.resize(number_of_layers_prc);
            for(int i = 0; i < number_of_layers_prc; ++i) 
            {
                if(i != 0) 
                {
                    prc[i].set(p_sizes[i - 1], p_sizes[i]);
                }
                else
                {
                    prc[i].set(first_in, p_sizes[i]);
                }
            }
        }
        
        /**Находит ошибку между значениями векторов.
        * @param rgb - матрица 1*n из изображения. 
        * @return - класс, к которому принадлежит объект.
        */
        int recognize(matrix rgb) 
        {
            matrix now = rgb;
            for(int i = 0; i < prc.size(); ++i) 
            {
                now = prc[i].predict(now);
            }
            int ans = 0;
            for(int i = 0; i < 10; ++i) 
            {
                if(now.get(0, i) > now.get(0, ans)) 
                {
                    ans = i;
                }
            }
            return ans;
        }


        void learn(matrix rgb, int ans, float learning_rate) {
            matrix target_output;
            target_output.set_size(1, 10);
            target_output.set(0, ans, 1);


            matrix in = rgb;
            for(int i = 0; i < prc.size(); ++i) {
                in = prc[i].predict(in);
            }
            matrix err = calculate_error(in, target_output);



            for (size_t i = prc.size() - 1; i > 0; --i)
            {
                prc[i].update_weights(err, 0.1);
                err = prc[i].calculate_hidden_error(err, prc[i - 1].get_weights());
            }
        }  

        int get_size() 
        {
            return number_of_layers_prc;
        }

        std::vector<std::vector<float>> ready_to_print_layer(int index) 
        {
            std::vector<std::vector<float>> ans(prc[index].get_size_x(), std::vector<float>(prc[index].get_size_y()));
            matrix weights = prc[index].get_weights();
            for(int i = 0; i < prc[index].get_size_x(); ++i) {
                for(int j = 0; j < prc[index].get_size_y(); ++j) {
                    ans[i][j] = weights.get(i, j);
                }
            }
            return ans;
        }

        void load(std::string filename) 
        {
            std::ifstream in;
            in.open(filename);
            int size;
            in >> size;
            prc.resize(size);
            for(int i = 0; i < size; ++i) {
                float size_x, size_y, value;
                in >> size_x >> size_y;
                matrix layer;
                layer.set_size(size_x, size_y);
                for(int x = 0; x < size_x; ++x) 
                {
                    for(int y = 0; y < size_y; ++y) 
                    {
                        in >> value;
                        layer.set(x, y, size_x);
                    }
                }
                prc[i].set_values(layer);
            }
        }
        
    private:
        int number_of_layers_prc;
        int learning_rate;
        std::vector<perceptronLayer> prc;
};


void save_NN(NN to_save, std::string filename) 
{
    std::ofstream out;
    out.open(filename);
    int size = to_save.get_size();
    out << size;
    
    for(int i = 0; i < size; ++i) 
    {
        std::vector<std::vector<float>> weights = to_save.ready_to_print_layer(i);
        out << weights.size() << ' ' << weights[0].size();
        for(int i = 0; i < weights.size(); ++i) 
        {
            for(int j = 0; j < weights[i].size(); ++j) 
            {
                out << weights[i][j] << ' ';
            }
            out << '\n';
        }
        out.close();
    }
}