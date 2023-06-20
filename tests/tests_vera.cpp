#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../nn/utility.cpp"
#include "doctest.h"


TEST_CASE("Проверка на равенство матриц")
{
    matrix m1(std::vector<std::vector<float>>{{2, 1},
                                              {2, 1}});
    CHECK(m1 == m1);
    matrix m2(std::vector<std::vector<float>>{{2, 1},
                                              {2, 1}});
    CHECK(m2 == m1);
    CHECK(m1 == m2);
    matrix m3(std::vector<std::vector<float>>{{3, 1},
                                              {2, 1}});
    CHECK(!(m3 == m2));
    CHECK(m1 != m3);
}

TEST_CASE("Общий функционал матриц")
{
    matrix m1(std::vector<std::vector<float>>{{2, 1},
                                              {2, 1}});
    matrix m2(std::vector<std::vector<float>>{{2, 1},
                                              {2, 1}});
    matrix m3(std::vector<std::vector<float>>{{3, 1},
                                              {2, 1}});
    matrix m4(std::vector<std::vector<float>>{{5, 6, 7}, {8, 9, 10}, {1, 2, 3}});
    matrix m5(std::vector<std::vector<float>>{{1, 1, 1}, {2, 3, 4}, {5, 1, 0}});

    matrix res4(std::vector<std::vector<float>>{{5, 6, 7}, {8, 9, 10}, {5, 2, 3}});

    m1.set(0, 0, 3);
    CHECK(m1 == m3);
    m1.add(0, 0, -1);
    CHECK(m1 == m2);
    CHECK(m1.get_size_x() == 2);
    CHECK(!(m1.get_size_x() == 3));

    CHECK(m1.get_size_y() == 2);
    CHECK(!(m1.get_size_y() == 3));

    CHECK(m4.get(0, 0) == 5 );
    CHECK(m4.get(1, 2) == 10 );
    CHECK(!(m1.get(0, 0) == 10 ));

    m4.set(2, 0, 5);
    CHECK(m4.get(2, 0) == 5);

    m5.set(0, 0, 10);
    CHECK(!(m5.get(0, 0) == 5));

    m5.add(2, 2, 10);
    CHECK(m5.get(2, 2) == 10);

    m4.add(0, 0, 10);
    CHECK(!(m4.get(0, 0) == 5));

    m4.set_size(3, 4);
    CHECK(m4.get_size_x() == 3);
    CHECK(m4.get_size_y() == 4);

    m5.set_size(1, 1);
    CHECK(!(m5.get_size_x() == 3));
    CHECK(!(m5.get_size_y() == 4));
}


TEST_CASE("Умножение матриц")
{
    matrix m1(std::vector<std::vector<float>> {{2, 1},{2, 1}});
    matrix m2(std::vector<std::vector<float>> {{3, 2}, {4, 5}});
    matrix res(std::vector<std::vector<float>> {{10, 5}, {18, 9}});


    CHECK(!(m1 * m2 == res));


    matrix m3(std::vector<std::vector<float>> {{1, 2, 3}});
    matrix m4(std::vector<std::vector<float>> {{5}, {6}, {7}});
    matrix res2(std::vector<std::vector<float>> {{38}});

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

TEST_CASE("Умножение матрицы на число") {
    matrix m1(std::vector<std::vector<float>> {{2, 1, 3}, {1, 2, 3}});
    matrix res1(std::vector<std::vector<float>> {{0,0,0},{0,0,0}});
    matrix res2(std::vector<std::vector<float>> {{1, 1, 1},{2, 2, 2}});
    float num = 0;
    matrix scalared = m1.scalar(num);

    CHECK(scalared == res1);
    CHECK(!(scalared == res2));
}


TEST_CASE("Транспонирование матрицы"){
    matrix m1(std::vector<std::vector<float>> {{1}, {1}});
    matrix res1(std::vector<std::vector<float>> {{1, 1}});

    matrix m2(std::vector<std::vector<float>> {{2, 3, 4}, {5, 6, 7}});
    matrix res2(std::vector<std::vector<float>> {{2, 3, 4}, {5, 6, 7}});

    matrix transposed = m1.transposeMatrix();
    matrix transposed2 = m2.transposeMatrix();

    CHECK(transposed == res1);
    CHECK(!(transposed2 == res2));
}

TEST_CASE("Проверка на квадрат"){
    matrix m1(std::vector<std::vector<float>> {{1, 2}, {1, 2}});
    matrix m2(std::vector<std::vector<float>> {{1, 2, 3}, {1, 2, 3}});

    CHECK(m1.isSquare());
    CHECK(!(m2.isSquare()));
}

TEST_CASE("Проверка на нулевность матрицы"){
    matrix m1(std::vector<std::vector<float>> {{0, 0}, {0, 0}});
    matrix m2(std::vector<std::vector<float>> {{1, 2, 3}, {1, 2, 3}});

    CHECK(m1.isZeroMatrix());
    CHECK(!(m2.isZeroMatrix()));
}
TEST_CASE("Расчет ошибки"){
    matrix out(std::vector<std::vector<float>> {{0, 0}, {0, 0}});
    matrix res(std::vector<std::vector<float>> {{1, 2}, {1, 2}});
    matrix calc = calc.calculate_error(out, res);

    matrix ress(std::vector<std::vector<float>> {{1, 4}});
    matrix ress2(std::vector<std::vector<float>> {{1, 9}});

    CHECK(calc == ress);
    CHECK(!(calc == ress2));
}

TEST_CASE("Активация"){
    matrix m1(std::vector<std::vector<float>> {{1, -100}, {1, 2}, {-1, -3}});
    matrix res = m1.activate_relu();

    matrix ress(std::vector<std::vector<float>> {{1, 0}, {1, 2}, {0, 0}});

    CHECK(res == ress);

}
TEST_CASE("Градиент"){
    matrix m1(std::vector<std::vector<float>> {{1, -100}, {1, 2}, {-1, -3}});
    matrix res = m1.gradients();

    matrix ress(std::vector<std::vector<float>> {{1, 0}, {1, 1}, {0, 0}});
    matrix ress2(std::vector<std::vector<float>> {{1, -100}, {100, 2}, {-100, -3}});

    CHECK(res == ress);
    CHECK(!(res == ress2));
}

TEST_CASE("Максимальное значение"){
    CHECK(max(1, 2, 3, 0) == 3);
    CHECK(!(max(1, 2, 3, 0) == 1));
}

TEST_CASE("Cумму элементов матрицы"){
    matrix m1(std::vector<std::vector<float>> {{1, 1}, {1, 2}, {1, 1}});

    CHECK(m1.sum() == 7);
    CHECK(!(m1.sum() == 8));
}
TEST_CASE("Среднее значение "){
    matrix m1(std::vector<std::vector<float>> {{2, 1}, {1, 2}, {1, 5}});

    CHECK(m1.mean() == 2);
    CHECK(!(m1.mean() == 500));

}
