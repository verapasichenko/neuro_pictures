#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../nn/utility.cpp"
#include "doctest.h"
#include <vector>

TEST_CASE("Проверка общей функциональности матриц") 
{
    matrix m1(std::vector<std::vector<float>> {{2, 1},{2, 1}});
    CHECK(m1 == m1);
    matrix m2(std::vector<std::vector<float>> {{2, 1},{2, 1}});
    CHECK(m2 == m1);
    CHECK(m1 == m2);
    matrix m3(std::vector<std::vector<float>> {{3, 1},{2, 1}});
    CHECK(m3 != m2);
    CHECK_NOT_EQ(m3 == m1);
    CHECK_NOT_EQ(m3 == m2);

    m1.set(0, 0, 3); 
    CHECK(m1 == m3);
    m1.add(0, 0, -1);
    CHECK(m1 == m2);
    CHECK(m1.get_size_x() == 2);
    CHECK(m1.get_size_y() == 2);
}


TEST_CASE("Умножение матриц") 
{
    matrix m1(std::vector<std::vector<float>> {{2, 1},{2, 1}});
    matrix m2(std::vector<std::vector<float>> {{3, 2}, {4, 5}});
    matrix res(std::vector<std::vector<float>> {{10, 9}, {2, 9}});

    CHECK(m1 * m2 == res);


    matrix m3(std::vector<std::vector<float>> {{1, 2, 3}});
    matrix m4(std::vector<std::vector<float>> {{5}, {6}, {7}});
    matrix res2(std::vector<std::vector<float>> {{18, 36, 54}});

    CHECK(m3 * m4 == res2);

    matrix m5(std::vector<std::vector<float>> {{2, 1, 3}, {1, 2, 3}});
    matrix m6(std::vector<std::vector<float>> {{7, 8}, {9, 10}, {11, 12}});
    matrix res3(std::vector<std::vector<float>> {{56, 62}, {58, 64}});

    CHECK(m5 * m6 == res3);
    
    matrix m7(std::vector<std::vector<float>> {{}});
    matrix m8(std::vector<std::vector<float>> {{}});
    matrix res4(std::vector<std::vector<float>> {{}});
    
    CHECK(m7 * m8 == res4);
}