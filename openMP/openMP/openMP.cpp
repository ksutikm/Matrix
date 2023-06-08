// openMP.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

void randomiseMatrix(int** m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = rand() % 100;
        }
    }
    return;
}

int main(int argc, char** argv) {
    srand(time(NULL));
    int n = 1500;
    int** matrix1;
    int** matrix2;
    matrix1 = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        matrix1[i] = (int*)malloc(sizeof(int) * n);
    }
    matrix2 = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        matrix2[i] = (int*)malloc(sizeof(int) * n);
    }
    randomiseMatrix(matrix1, n);
    randomiseMatrix(matrix2, n);

    int** result = (int**)malloc(sizeof(int*) * n);;
    for (int i = 0; i < n; i++) {
        result[i] = (int*)malloc(sizeof(int) * n);
    }

    cout << "threads,runtime " << endl;
    for (int t = 1; t < 6; t++)
    {
        int threadsNum = t;
        omp_set_num_threads(threadsNum);
        int i, j, k;
        unsigned int start_time = clock();
        #pragma omp parallel for private(i, j, k)
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                result[i][j] = 0;
                for (k = 0; k < n; k++) {
                    result[i][j] += (matrix1[i][k] * matrix2[k][j]);
                }
            }
        }
        unsigned int end_time = clock();
        cout <<t << "," << (end_time - start_time) / 1000 << endl;

    }
    return 0;
}
