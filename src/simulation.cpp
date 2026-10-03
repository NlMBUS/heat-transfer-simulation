#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;
//set PATH=C:\Program Files\OpenCV\x64\mingw\bin;%PATH% && g++ simulation.cpp -I"C:\Program Files\OpenCV\include" -L"C:\Program Files\OpenCV\x64\mingw\lib" -lopencv_core455 -lopencv_imgcodecs455 -lopencv_highgui455 -o main && main
float epsilon = 0.01;
//2858
void pixel_init(int y, int x, vector<vector<double>>& grid, Mat img) {
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            if (j >= x/3 && j <= 1+x*2/3 && i <= 1+y/12) {//set heater temp and colour
                grid[i][j] = 100;
                img.data[i * img.step[0] + j * img.step[1]] = 0;
                img.data[i * img.step[0] + j * img.step[1] + 1] = 0;
                img.data[i * img.step[0] + j * img.step[1] + 2] = 255;
            }
            else if (j <= x/20 || j >= x*19/20 || i <= y/20 || i >= y*19/20) {//set wall temp and colour
                grid[i][j] = 20;
                img.data[i * img.step[0] + j * img.step[1]] = 171;
                img.data[i * img.step[0] + j * img.step[1] + 1] = 255;
                img.data[i * img.step[0] + j * img.step[1] + 2] = 171;
            }
            else {//set room temp
                grid[i][j] = -20;
            }
        }
    }
}

bool pixel_update(int y, int x, vector<vector<double>>& grid, Mat img) {
    vector<vector<double>> updated(y, vector<double>(x, 0));
    pixel_init(y, x, updated, img);
    float maxChange = 0;
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            float prev = grid[i][j];
            if ((j > x/3 && j < x*2/3 && i < y/12)  || (j < x/20 || j > x*19/20 || i < y/20 || i > y*19/20)) {//keep heater and wall temp/colour constant by skipping
                continue;
            }
            updated[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1])/4;//changes temperature by having each pixel take average of colours and temps around it
            if (updated[i][j] < 40) {//colour changing when cold
                img.data[i * img.step[0] + j * img.step[1]] = (20+updated[i][j])/60*255;
                img.data[i * img.step[0] + j * img.step[1] + 1] = 255;
                img.data[i * img.step[0] + j * img.step[1] + 2] = (20+updated[i][j])/60*255;
            } else if (updated[i][j] > 40){//colour changing when hot
                img.data[i * img.step[0] + j * img.step[1]] = (-updated[i][j]+100)/60*255;
                img.data[i * img.step[0] + j * img.step[1] + 1] = (-updated[i][j]+100)/60*255;
                img.data[i * img.step[0] + j * img.step[1] + 2] = 255;
            } else {//average temperature is white
                img.data[i * img.step[0] + j * img.step[1]] = 255;
                img.data[i * img.step[0] + j * img.step[1] + 1] = 255;
                img.data[i * img.step[0] + j * img.step[1] + 2] = 255;                
            }
            float diff = fabs(grid[i][j] - updated[i][j]);
            if (diff > maxChange) {
                maxChange = diff;
            }
        }
    }
    grid = updated;
    return (maxChange < epsilon);//if temperature is changing by less than epsilon then signal to stop
}

int main(int argc, char* argv[]) {
    int y, x;
    Mat img;
    if (argc < 3) {
        printf("Provide two arguments:\n");
        printf("- number of rows\n");
        printf("- number of columns\n");
        exit(1);
    }
    y = atoi(argv[1]);
    x = atoi(argv[2]);
    img = Mat(y, x, CV_8UC3, Scalar(0, 255, 0));
    vector<vector<double>> temps(y, vector<double>(x, 0));
    pixel_init(y, x, temps, img);
    int i = 0;
    while (true) {//continuous loop that breaks if function returns true
        if (pixel_update(y, x, temps, img)){
            break;
        }
        imshow("heat distribution", img);
        if (waitKey(10) >= 0) {
            break;
        }
        cout << i << endl;
        i++;
    }
    return 0;
}
