#include <iostream>
#include <fstream>
#include <string>
#include "nn/network.cpp"


int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


int main() {
    std::string dataset_path, dataset_ans_path;
    std::cin >> dataset_path >> dataset_ans_path;
    std::ifstream file, in;
    in.open(dataset_ans_path);
    file.open(dataset_path);  
    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    int k;

    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    file.read((char*)&n_cols,sizeof(n_cols));
    n_rows = reverseInt(n_rows);
    n_cols= reverseInt(n_cols);

    NN recognizer(3, {500, 20, 10}, n_rows * n_cols); 
    
    unsigned char ns = 0;
    for(int i = 0; i < 8; ++i) 
    {
        in.read((char*)&ns,sizeof(ns));
    }   
    for(int i = 0; i < number_of_images; ++i)
    {
        matrix image(std::vector<std::vector<float>> (1, std::vector<float>(n_cols * n_rows, 0)));

        for(int r = 0; r < n_rows; ++r)
        {
            for(int c = 0; c < n_cols; ++c)
            {
                unsigned char temp = 0;
                file.read((char*)&temp,sizeof(temp));
                image.set(0, r * n_rows + c, temp);
            }
        } 
        unsigned char ns = 0;
        in.read((char*)&ns,sizeof(ns));
        int smth = ns;
        recognizer.learn(image, smth, 0.1);
        std::cout << smth << ' ' << i << '\n';
        if(i % 500 == 0) 
        {
            save_NN(recognizer, "network.txt");
        }
    }
}