#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
using namespace std;
using namespace cv;

// UtilityFunctions 类提供计算图像平均饱和度和亮度的静态方法
class UtilityFunctions {
public:
    // 计算并返回图像的平均饱和度和亮度
    static pair<double, double> calculateAverageSaturationAndBrightness(const Mat& frame) {
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        double sumSaturation = 0;
        double sumBrightness = 0;
        int pixelCount = hsv.rows * hsv.cols;
        for (int row = 0; row < hsv.rows; row++) {
            for (int col = 0; col < hsv.cols; col++) {
                Vec3b hsvPixel = hsv.at<Vec3b>(row, col);
                sumSaturation += hsvPixel[1];
                sumBrightness += hsvPixel[2];
            }
        }
        return make_pair(sumSaturation / pixelCount, sumBrightness / pixelCount);
    }
};

// detectEdges 函数用于检测图像中的边缘
Mat detectEdges(Mat& frame) {
    Mat gray, blurred, edges;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(9, 9), 2.0);
    Canny(blurred, edges, 40, 120, 3);
    return edges;
}

// LampBar 类表示一个灯条，包含面积、中心坐标等属性
class LampBar {
public:
    double area;
    double x;
    double y;
    Rect boundingBox; // 添加最小外接矩形属性
    // 返回灯条的中心点坐标
    Point center() const { return Point(static_cast<int>(x), static_cast<int>(y)); }
    LampBar(double area, double x, double y, Rect boundingBox)
        : area(area), x(x), y(y), boundingBox(boundingBox) {}
};

// balanceFrame 函数用于调整图像的色调，以便于处理
Mat balanceFrame(Mat& frame) {
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    vector<Mat> hsvChannels;
    split(hsv, hsvChannels);
    equalizeHist(hsvChannels[2], hsvChannels[2]);
    merge(hsvChannels, hsv);
    return hsv;
}

// isBrighterAndMoreSaturatedThanAverage 函数用于判断像素点是否比平均值更亮更饱和
bool isBrighterAndMoreSaturatedThanAverage(const Mat& frame, const Point& point, double averageSaturation, double averageBrightness) {
    Vec3b pixel = frame.at<Vec3b>(point);
    double saturation = pixel[1];
    double brightness = pixel[2];
    if (((saturation - averageSaturation)/averageSaturation > 0.3) && ((brightness - averageBrightness)/averageBrightness > 0.3)) {
        return true;
    }
    return false;
}

// ContourFilter 类提供筛选轮廓的静态方法
class ContourFilter {
public:
    // 根据轮廓的面积、周长、长宽比等条件筛选轮廓
    static vector<vector<Point>> filterContours(const Mat& frame, const vector<vector<Point>>& contours) {
        vector<vector<Point>> filteredContours;
        vector<double> perimeters;
        vector<Rect> boundingBoxes(contours.size());
        vector<double> aspectRatios(contours.size());
        if (contours.empty()) {
            return filteredContours;
        }

        // 计算每个轮廓的周长、外接矩形以及长宽比
        for (size_t j = 0; j < contours.size(); j++) {
            double perimeter = arcLength(contours[j], true);
            perimeters.push_back(perimeter);
            boundingBoxes[j] = boundingRect(contours[j]);
            aspectRatios[j] = (double)boundingBoxes[j].height / (boundingBoxes[j].width + 1e-6);
        }

        // 根据条件筛选轮廓
        for (size_t j = 0; j < contours.size(); j++) {
            vector<Point> approx;
            double epsilon = 0.02 * perimeters[j];
            approxPolyDP(Mat(contours[j]), approx, epsilon, true);

            double area = contourArea(contours[j]);
            if ((area > 25 && area < 500 && aspectRatios[j] >= 1.2 && aspectRatios[j] <= 3.5) ||
                (area >= 500 && aspectRatios[j] >= 2 && aspectRatios[j] <= 6) &&
                (approx.size() >= 3 && approx.size() <= 8)) {
                filteredContours.push_back(contours[j]);
            }
        }

        return filteredContours;
    }
};

// matchLampBars 函数用于在图像中匹配灯条，并返回筛选后的轮廓的最小外接矩形
vector<LampBar> matchLampBars(Mat& frame) {
    Mat frame_hsv = balanceFrame(frame);
    GaussianBlur(frame_hsv, frame_hsv, Size(3, 3), 1, 1);
    Mat edges = detectEdges(frame);
    pair<double, double> avgSaturationBrightness = UtilityFunctions::calculateAverageSaturationAndBrightness(frame);
    double averageSaturation = avgSaturationBrightness.first;
    double averageBrightness = avgSaturationBrightness.second;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> filteredContours = ContourFilter::filterContours(edges, contours);

    vector<LampBar> lamps;
    for (size_t i = 0; i < filteredContours.size(); i++) {
        double area = contourArea(filteredContours[i]);
        if (area > 20.0) {
            Moments M = moments(filteredContours[i]);
            int x = static_cast<int>(M.m10 / M.m00 + 0.5);
            int y = static_cast<int>(M.m01 / M.m00 + 0.5);
            Point center = Point(x, y);
            Rect boundingBox = boundingRect(filteredContours[i]);
            if (isBrighterAndMoreSaturatedThanAverage(frame, center, averageSaturation, averageBrightness)) {
                lamps.push_back(LampBar(area, x, y, boundingBox));
            }
        }
    }
    return lamps;
}

// drawLampLines 函数用于绘制灯条和它们的最小外接矩形
void drawLampLines(Mat& frame, const vector<LampBar>& lamps) {
    Mat drawing = frame.clone();
    Scalar rectColor = Scalar(0, 0, 255); // 最小外接矩形的颜色设置为红色
    Scalar centerColor = Scalar(0, 255, 0); // 灯条中心点的颜色设置为绿色
    namedWindow("灯条检测", WINDOW_AUTOSIZE);

    // 绘制灯条中心和最小外接矩形
    for (const auto& lamp : lamps) {
        circle(drawing, lamp.center(), 3, centerColor, -1); // 绘制灯条中心点
        rectangle(drawing, lamp.boundingBox, rectColor, 2); // 绘制最小外接矩形
    }
    for (size_t i = 0; i < lamps.size(); i++) {
        for (size_t j = i + 1; j < lamps.size(); j++) {
            double areaDiffRatio = fabs((lamps[i].area - lamps[j].area) / lamps[i].area);
            double heightDiff = fabs(lamps[i].y - lamps[j].y);
            double avgArea = (lamps[i].area + lamps[j].area) / 2;
            double heightDiffThreshold = pow(avgArea, 0.1);
            if (heightDiff < heightDiffThreshold) {
                line(drawing, lamps[i].center(), lamps[j].center(), Scalar(0, 0, 255), 2);
            }
        }
    }
    imshow("灯条检测", drawing);
}

int main() {
    Mat frame;
    VideoCapture capture("/home/linux/Downloads/test2.mp4");
    if (!capture.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return -1;
    }

    while (true) {
        capture >> frame;
        if (frame.empty()) {
            break;
        }
        vector<LampBar> lamps = matchLampBars(frame); // 匹配灯条
        drawLampLines(frame, lamps); // 绘制灯条和最小外接矩形

        if (waitKey(30) >= 0) break;
    }

    return 0;
}