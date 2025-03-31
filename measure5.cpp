#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <tuple>
#include <sstream>
#include <functional>
#include <vector>
#include <limits>


int click = 0;
int term = 0;
cv::Point2f point1, point2, point3, point4, point5;
double a0, a1, a2, a3, a4; 
double phi;
std::vector<double> phi_list;
double phi_rms, phi_sigma, theta_0;
std::string coo;
std::ostringstream phi_text, theta_text, phi_rms_text, phi_sigma_text, theta_0_text;
std::ostringstream beta_text, pb_ele_mc_text, bpc_text;
std::ostringstream point_text;
double cm = 53.69; //ピクセル

int n = 5; //phiの測定数
std::string name = "example2"; //読み込むファイルの名前
std::string file = "scatter26"; //書き出すファイルの名前に付け加えるところ

int m = 15; //文字の列の高さ指定
int height = (n+12)*m; //(n+n+2) > (n+10)だったら、(n+n+4)*mにする
std::vector<cv::Point> points2;
std::vector<cv::Point> result;

int x_length = 400; //デフォルト値
int x_1 = 100;
int x_2 = 200;
int x_3 = 300;
int x_4 = 400;
int x_5 = 500;
int x_6 = 600;
int x_7 = 800;
int x_8 = 1000;
int x_9 = 1200;


///////////////////////////////////////////////////////////////////////////////
//ある長さの線を引く(distance関係なし)
double draw_line(cv::Mat& img, cv::Point p1, cv::Point p2, int length, cv::Scalar color, int width) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    cv::Point p_end = p1;

    if (distance > 0) {
        double scale = length / distance;
        p_end = cv::Point(p1.x + static_cast<int>(dx * scale), p1.y + static_cast<int>(dy * scale));
        cv::line(img, p1, p_end, color, width);
    }

    double angle = std::atan2(-dy, dx);
    return angle * 180.0 / M_PI;
}

//ある長さの線を引く(distanceぶん)
double draw_line2(cv::Mat& img, cv::Point p1, cv::Point p2, cv::Scalar color, int width){
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    cv::Point p_end = p1;

    if (distance > 0) {
        double scale = 1;
        p_end = cv::Point(p1.x + static_cast<int>(dx * scale), p1.y + static_cast<int>(dy * scale));
        cv::line(img, p1, p_end, color, width);
    }

    double angle = std::atan2(-dy, dx);
    return angle * 180.0 / M_PI;
}

//ある長さの線を引く(distance+length)　+n分割のところに点を打つ
std::tuple<std::vector<cv::Point>, double> draw_line_n(cv::Mat& img, cv::Point p1, cv::Point p2, int length, cv::Scalar color, int width) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    std::vector<cv::Point> points(n+1);
    points[0] = p1; //使わないけど、とりあえずiの数字合わせでいれておく

    if (distance > 0) {
        for (int i = 1; i<=n; i++) {
            double scale = 1 + length/distance * i/static_cast<double>(n);
            points[i] = cv::Point(p1.x + static_cast<int>(dx*scale), p1.y + static_cast<int>(dy*scale));
        }
        cv::line(img, p1, points[n], color, width);
    }
    double angle = std::atan2(-dy, dx);
    return std::make_tuple(points, angle * 180.0 / M_PI);
}


//変数の取り出し方
//std::tie(p_med, p_end) = draw_line3(...)

//ある長さの線を引く(distance+lengthぶん)
double draw_line4(cv::Mat& img, cv::Point p1, cv::Point p2, int length, cv::Scalar color, int width){
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    cv::Point p_end = p1;

    if (distance > 0) {
        double scale = 1 + length/distance;
        p_end = cv::Point(p1.x + static_cast<int>(dx * scale), p1.y + static_cast<int>(dy * scale));
        cv::line(img, p1, p_end, color, width);
    }

    double angle = std::atan2(-dy, dx);
    return angle * 180.0 / M_PI;
}


//ある長さの垂直な線を引く(distance関係なし)
double draw_subline(cv::Mat& img, cv::Point p_int, cv::Point p1, cv::Point p2, int length, cv::Scalar color, int width) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    double dx_ver = -dy;
    double dy_ver = dx;
    cv::Point p_end = p_int;

    if (distance > 0) {
        double scale = length / distance;
        p_end = cv::Point(p_int.x + static_cast<int>(dx_ver * scale), p_int.y + static_cast<int>(dy_ver * scale));
        cv::line(img, p_int, p_end, color, width);
    }
    double angle = std::atan2(-dy_ver, dx_ver);
    return angle * 180.0 / M_PI;
}

//始点を指定し、p1からp2までの長さの線を引く。長方形用。
void draw_rectangle(cv::Mat& img, cv::Point p_1, cv::Point p1, cv::Point p2, cv::Point p3, cv::Point p4, cv::Scalar color, int width) {
    double dx_v = p2.x - p1.x;
    double dy_v = p2.y - p1.y;
    double dx_h = p4.x - p3.x;
    double dy_h = p4.y - p3.y;
    cv::Point p_2 = p_1;
    cv::Point p_3 = p_1;
    cv::Point p_4 = p_1;
    double distance = std::sqrt(dx_v*dx_v + dy_v*dy_v);

    if (distance > 0) {
        double scale = 1;
        p_2 = cv::Point(p_1.x + static_cast<int>(dx_v), p_1.y + static_cast<int>(dy_v));
        p_4 = cv::Point(p_1.x + static_cast<int>(dx_h), p_1.y + static_cast<int>(dy_h));
        p_3 = cv::Point(p_4.x + static_cast<int>(dx_v), p_4.y + static_cast<int>(dy_v));
        cv::line(img, p_1, p_2, color, width);
        cv::line(img, p_2, p_3, color, width);
        cv::line(img, p_3, p_4, color, width);
        cv::line(img, p_4, p_1, color, width);
    }
}



//////////////////////////////////////////////////////////////////////////
//ニュートン法の関数
double NewtonRaphson(std::function<double(double)> func, std::function<double(double)> dfunc, double initial_guess, double tol = 1e-6, int max_iter = 100) {
    double x = initial_guess;
    for (int i = 0; i < max_iter; i++) {
        double fx = func(x);
        double dfx = dfunc(x);
        if (std::fabs(dfx) < 1e-8) { // ゼロ割防止
            std::cerr << "Derivative too small, stopping iteration" << std::endl;
            return x;
        }
        double x_new = x - fx / dfx;
        if (std::fabs(x_new - x) < tol) return x_new;
        x = x_new;
    }
    std::cerr << "Newton-Raphson method did not converge" << std::endl;
    return x;
}

// 数値微分（中央差分）
double numerical_derivative(std::function<double(double)> func, double x, double h = 1e-6) {
    return (func(x + h) - func(x - h)) / (2 * h);
}

// fsolve に近い非線形方程式の解法（数値微分付き Newton-Raphson + 二分法）
double fsolve(std::function<double(double)> func, double initial_guess, double tol = 1e-6, int max_iter = 100) {
    double x = initial_guess;
    for (int i = 0; i < max_iter; i++) {
        double fx = func(x);
        double dfx = numerical_derivative(func, x);
        
        if (std::fabs(dfx) < 1e-8) {
            std::cerr << "Derivative too small, switching to bisection." << std::endl;
            x *= 0.9; // 少し減らして収束しやすくする
            continue;
        }

        double x_new = x - fx / dfx;
        if (std::fabs(x_new - x) < tol) return x_new;
        x = x_new;
    }
    std::cerr << "fsolve method did not converge" << std::endl;
    return x;
}


// f1(beta) の関数
// double f1(double beta, double x, double theta, double mc2) {
//     const double z = 1.0;
//     const double X0 = 30050.0;
//     return 13.6 / mc2 * std::sqrt(1 - beta * beta) / (beta * beta) * z * std::sqrt(x / X0) *(1 + 0.038 * std::log(x * z * z / (X0 * beta * beta))) - theta;
// }


double f1(double beta, double x, double theta, double mc2) {
    const double z = 1.0;
    const double X0 = 30050.0;
    double log_term = std::log(std::max(x * z * z / (X0 * beta * beta), 1e-8));
    return 13.6 / mc2 * std::sqrt(std::max(1 - beta * beta, 1e-8)) / std::max(beta * beta, 1e-8) * z * std::sqrt(x / X0) * (1 + 0.038 * log_term) - theta;
}

// f1(beta) の微分
double df1(double beta, double x, double mc2) {
    const double z = 1.0;
    const double X0 = 30050.0;    
    double term1 = -13.6 / mc2 * z * std::sqrt(x / X0);
    double term2 = (1 + 0.038 * std::log(x * z * z / (X0 * beta * beta)));
    double term3 = -2 * beta * std::sqrt(1 - beta * beta) / (beta * beta * beta);
    double term4 = -0.076 / beta;

    return term1 * (term2 * term3 + term4);
}
// f2(bpc) の関数
double f2(double bpc, double x, double theta) {
    const double z = 1.0;
    const double X0 = 30050.0;
    double beta = 1.0;
    return 13.6 / bpc * z * std::sqrt(x / X0) * (1.0 + 0.038 * std::log(x * z * z / (X0 * beta * beta))) - theta;
}
// f2(bpc) の微分
double df2(double bpc, double x) {
    double beta = 1.0;
    const double X0 = 30050.0;
    const double z = 1.0;
    return -13.6 / (bpc * bpc) * z * std::sqrt(x / X0) * (1.0 + 0.038 * std::log(x * z * z / (X0 * beta * beta)));
}

/////////////////////////////////////////////////////////////////////////////

//x_lengthの決定用
void set_x_length(int key) {
    switch (key) {
        case '1': x_length = x_1; break;
        case '2': x_length = x_2; break;
        case '3': x_length = x_3; break;
        case '4': x_length = x_4; break;
        case '5': x_length = x_5; break;
        case '6': x_length = x_6; break;
        case '7': x_length = x_7; break;
        case '8': x_length = x_8; break;
        case '9': x_length = x_9; break;
        default: break;
    }
}

//ちょっと名前とか変えた
///////////////////////////////////////////////////////////////////////////////
cv::Mat img, img_display, img_color;
float scale = 1.0f;
cv::Point2f offset(0, 0);
const float scale_factor = 1.1f;  // 拡大率
const int move_step = 20;         // 移動量
const float min_scale = 1.0f;     // 最小拡大率
const float max_scale = 5.0f;     // 最大拡大率


//clickした画面上の点を画像上の点に変換する関数
cv::Point2f tp(cv::Point point) {
    return cv::Point2f((point.x + offset.x)/scale, (point.y + offset.y)/scale);
}
double tpx(int x) {
    return (x+offset.x)/scale;
}
double tpy(int y) {
    return (y+offset.y)/scale;
}



void update_display() {
    cv::Mat img_resized;
    cv::resize(img_color, img_resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, -offset.x, 0, 1, -offset.y);
    cv::warpAffine(img_resized, img_display, M, img.size());
    cv::imshow("img_bin", img_display);
}

void on_key(int key) {
    std::ostringstream name_in, name_out;
    name_in << "./" << name << ".png";
    name_out << "./png_test/" << name << "_" << file <<".png";


    switch (key) {
        case 59:  // '+' で拡大
            if (scale * scale_factor <= max_scale) scale *= scale_factor;
            break;
        case 45:  // '-' で縮小
            if (scale / scale_factor >= min_scale) scale /= scale_factor;
            break;
        case 82:  // ↑ //ちゃんと制限かかってる
            if (offset.y - move_step >= 0) 
                offset.y -= move_step;
            break;
        case 84:  // ↓ //ちゃんと制限かかってる
            if (offset.y + move_step + img.rows < img.rows*scale)
                offset.y += move_step;
            break;
        case 81:  // ←　//ちゃんと制限かかってる
            if (offset.x - move_step >= 0) 
                offset.x -= move_step;
            break;
        case 83:  // → //制限かかってない
            if (offset.x + move_step + img.cols < img.cols*scale)
                offset.x += move_step;
            break;
            
        case 's':
            cv::imwrite(name_out.str(), img_color);  // 画像を保存
            break;
        case 27:  // ESC で終了
            exit(0);
    }
    update_display();
}


//全部一旦point2とかで取っておいて、そこからさらにその時点でtpしたやつを作っておく。全部そっちで書き直す
//lineとかに入れるときにtransformするのがいい <- これだとずれる
//この方法なら、click間で動いても大丈夫。ガイドラインの選択までの間には動けない
//ガイドラインの選択とかも全部clickで分ければ、pointも維持できるし、1手戻るとかの操作もできそう??
/////////////////////////////////////////////////////////////////////////////////
void at_click(int event, int x, int y, int flags, void* param) {
    if (event == cv::EVENT_LBUTTONDOWN) {  // 左クリック時
        click++;
        cv::Mat* img = static_cast<cv::Mat*>(param);
        double x_cm = x_length/cm;
        

        if (click == 1) {
            //point1の変換
            point1 = tp(cv::Point(x,y));

            //座標取り出し
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "1 (" << std::fixed << std::setprecision(2) << point1.x << ", " << point1.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*0+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
            std::cout << point_text.str() << std::endl;

            //変換point1で
            cv::circle(*img, point1, 4, cv::Scalar(0, 0, 255), -1);
            cv::putText(*img, "1", cv::Point2f(point1.x + 5, point1.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,255), 1.5);
            
            //ここはscaleなどの影響受けない
            draw_rectangle(*img, cv::Point(img->cols - 430, 20), cv::Point(0,0), cv::Point(300,0), cv::Point(0,0), cv::Point(0,40), cv::Scalar(255,0,0),2);
            cv::putText(*img, "select x_length", cv::Point(img->cols - 400, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);
            update_display();
            while (true) {
                int key = cv::waitKey(0);  // キー入力を待つ
                if (key == '1' || key == '2' || key == '3' || key == '4' || key == '5' || key == '6' || key == '7' || key == '8' || key == '9') {
                    set_x_length(key);
                    break;
                }
            }
            cv::putText(*img, "x  " + std::to_string(x_length), cv::Point(230, 50+0*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            x_cm = x_length/cm;
            cv::putText(*img, "x cm " + std::to_string(x_cm), cv::Point(230, 50+1*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
        


            //ガイドライン1を消す

            draw_rectangle(*img, cv::Point(img->cols - 430, 20), cv::Point(0,0), cv::Point(300,0), cv::Point(0,0), cv::Point(0,40), cv::Scalar(100,100,100),2);
            cv::putText(*img, "select x_length", cv::Point(img->cols - 400, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(100,100,100), 2);
            //ガイドライン2
            cv::circle(*img, point1, x_length/10, cv::Scalar(255, 0, 0), 1); //幅20の円を書く


            ///////////////////////たぶんここまでok
            


            


        } else if (click == 2) {
            //＊step1
            point2 = tp(cv::Point(x,y));
            //座標取り出し
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "2 (" << std::fixed << std::setprecision(2) << point2.x << ", " << point2.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*1+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
            std::cout << point_text.str() << std::endl;

            cv::circle(*img, point2, 4, cv::Scalar(0, 0, 255), -1);
            cv::putText(*img, "2", cv::Point2f(point2.x + 5, point2.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,255), 1.5);
            //ガイドライン2を消す
            cv::circle(*img, point1, x_length/10, cv::Scalar(100, 100, 100), 1); //幅20の円を書く
            

            //＊step2
            auto [result, angle] = draw_line_n(*img, point1, point2, x_length, cv::Scalar(100, 100, 255), 1);
            //std::tie(point2_1, point2_2, point2_3, ..., point2_n, a0) = draw_line3(*img, point1, point2, x_length, cv::Scalar(100, 100, 255), 1);
            //points2は、クリックで決めるわけじゃないからずれない、scale, offsetの影響受けない
            points2 = result;
            for (int i = 1; i<=n; i++){
                cv::circle(*img, points2[i], 4, cv::Scalar(0,0,255), -1);
                //座標取り出し
                point_text.str("");  // 文字列の内容をリセット
                point_text.clear();  // 状態フラグをリセット
                point_text << "2_" << std::to_string(i) << " (" << std::fixed << std::setprecision(2) << points2[i].x << ", " << points2[i].y << ")";
                cv::putText(*img, point_text.str(), cv::Point(20, 50+m*(1+i)+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
                std::cout << point_text.str() << std::endl;
                cv::putText(*img, "2_"+std::to_string(i), cv::Point(points2[i].x + 5, points2[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,255), 1.5);
            }



            ////////////////////////////////////////////////ここまでok


            //＊step3
            draw_subline(*img, points2[n], point1, point2, 20, cv::Scalar(0, 255, 0), 1); //key 1の向き
            draw_subline(*img, points2[n], point1, point2, -20, cv::Scalar(255, 0, 0), 1); //key 2の向き
            cv::putText(*img, "x  " + std::to_string(x_length), cv::Point(point2.x + 50, point2.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,255,0), 1);

            //＊step4
            update_display();
            int key = cv::waitKey(0);  // キー入力を待つ、ガイドラインの色を消す
            draw_subline(*img, points2[n], point1, point2, 20, cv::Scalar(100,100,100), 1); //key 1の向き
            draw_subline(*img, points2[n], point1, point2, -20, cv::Scalar(100,100,100), 1); //key 2の向き
            if (key == '1') {  // 1が押されたら、point1->point2に対して右側に線を引く
                for (int i=1; i<=n; i++){
                    draw_subline(*img, points2[i], point1, point2, x_length, cv::Scalar(100, 100, 100), 1);
                    draw_subline(*img, points2[i], point1, point2, -x_length/10, cv::Scalar(100, 100, 100), 1);
                }
            } else if (key == '2') {  // 2が押されたら、point1->point2に対して左側に線を引く
                for (int i=1; i<=n; i++){
                    draw_subline(*img, points2[i], point1, point2, -x_length, cv::Scalar(100, 100, 100), 1);
                    draw_subline(*img, points2[i], point1, point2, x_length/10, cv::Scalar(100, 100, 100), 1);
                }
            }

            //ガイドライン3
            cv::circle(*img, points2[n], 4, cv::Scalar(255,0,0), -1);
            draw_subline(*img, points2[n], point1, point2, 10, cv::Scalar(255, 0, 0), 1); //key 1の向き
            draw_subline(*img, points2[n], point1, point2, -10, cv::Scalar(255, 0, 0), 1); //key 2の向き


            //////////////////////////ここまでok



        } else if (click == 3) {
            point3 = tp(cv::Point(x,y));
            cv::circle(*img, point3, 4, cv::Scalar(0, 0, 255), -1);
            cv::putText(*img, "3", cv::Point2f(point3.x + 5, point3.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,255), 1.5);
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "3 (" << std::fixed << std::setprecision(2) << point3.x << ", " << point3.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*(n+2)+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
            std::cout << point_text.str() << std::endl;
            

            draw_line2(*img, points2[n], point3, cv::Scalar(100, 100, 255), 1);


            //ガイドライン3を消す
            cv::circle(*img, points2[n], 4, cv::Scalar(0,0,255), -1);
            draw_subline(*img, points2[n], point1, point2, 10, cv::Scalar(100,100,100), 1); //key 1の向き
            draw_subline(*img, points2[n], point1, point2, -10, cv::Scalar(100,100,100), 1); //key 2の向き

            //ガイドライン4を書く
            cv::circle(*img, points2[1], 4, cv::Scalar(255,0,0), -1);
            draw_subline(*img, points2[1], point1, point2, 10, cv::Scalar(255,0,0), 1); //key 1の向き
            draw_subline(*img, points2[1], point1, point2, -10, cv::Scalar(255,0,0), 1); //key 2の向き

            /////////////////////////////////ここまでok




        } else if (click >= 3+1 && click <= 3+n-2) {
            int j = click - 3; //point4の番号、points2[j]に対応する
            //＊step1
            point4 = tp(cv::Point(x,y));
            cv::circle(*img, point4, 4, cv::Scalar(0, 0, 255), -1);
            cv::putText(*img, "4_" + std::to_string(j), cv::Point2f(point4.x + 5, point4.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,255), 1.5);

            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "4_" << std::to_string(j) << " (" << std::fixed << std::setprecision(2) << point4.x << ", " << point4.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*(n+2+j)+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
            std::cout << point_text.str() << std::endl;


            //＊step2
            draw_line2(*img, points2[j], point4, cv::Scalar(100, 100, 100), 1);
            a2 = draw_line2(*img, point4, point3, cv::Scalar(100, 100, 255), 1);
            a3 = draw_line4(*img, point2, point4, x_length/10, cv::Scalar(100, 100, 255), 1);
            phi = a2-a3;
            phi_list.push_back(phi);

            std::cout << a2 << std::endl;
            std::cout << a3 << std::endl;
            std::cout << "phi"<< j << ": " << a2-a3 << std::endl;


            phi_text.str("");  // 文字列の内容をリセット
            phi_text.clear();  // 状態フラグをリセット
            phi_text << "phi" << j << " " << std::fixed << std::setprecision(2) << (a2 - a3);
            cv::putText(*img, phi_text.str(), cv::Point(point4.x + 20, point4.y + 50), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,255,0), 1);
            cv::putText(*img, phi_text.str(), cv::Point(230, 50+(j+1)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);


            //ガイドライン4を消す
            cv::circle(*img, points2[j], 4, cv::Scalar(0,0,255), -1);
            draw_subline(*img, points2[j], point1, point2, 10, cv::Scalar(100,100,100), 1); //key 1の向き
            draw_subline(*img, points2[j], point1, point2, -10, cv::Scalar(100,100,100), 1); //key 2の向き
            //ガイドライン5を書く
            cv::circle(*img, points2[j+1], 4, cv::Scalar(255,0,0), -1);
            draw_subline(*img, points2[j+1], point1, point2, 10, cv::Scalar(255,0,0), 1); //key 1の向き
            draw_subline(*img, points2[j+1], point1, point2, -10, cv::Scalar(255,0,0), 1); //key 2の向き

            ////////////////////////////ここまでok


        } else if (click == 3+n-1){
            int j = click - 3; //point4の番号、points2[j]に対応する
            //＊step1
            point4 = tp(cv::Point(x,y));
            cv::circle(*img, point4, 4, cv::Scalar(0, 0, 255), -1);
            cv::putText(*img, "4_" + std::to_string(j), cv::Point2f(point4.x + 5, point4.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,255), 1.5);

            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "4_" << std::to_string(j) << " (" << std::fixed << std::setprecision(2) << point4.x << ", " << point4.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*(n+2+n-1)+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
            std::cout << point_text.str() << std::endl;


            //＊step2
            draw_line2(*img, points2[j], point4, cv::Scalar(100, 100, 100), 1);
            a2 = draw_line2(*img, point4, point3, cv::Scalar(100, 100, 255), 1);
            a3 = draw_line4(*img, point2, point4, x_length/10, cv::Scalar(100, 100, 255), 1);
            phi = a2-a3;
            phi_list.push_back(phi);

            std::cout << a2 << std::endl;
            std::cout << a3 << std::endl;
            std::cout << "phi"<< j << ": " << a2-a3 << std::endl;


            phi_text.str("");  // 文字列の内容をリセット
            phi_text.clear();  // 状態フラグをリセット
            phi_text << "phi" << j << " " << std::fixed << std::setprecision(2) << (a2 - a3);
            cv::putText(*img, phi_text.str(), cv::Point(point4.x + 20, point4.y + 50), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,255,0), 1);
            cv::putText(*img, phi_text.str(), cv::Point(230, 50+(j+1)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);


            //ガイドライン4を消す
            cv::circle(*img, points2[j], 4, cv::Scalar(0,0,255), -1);
            draw_subline(*img, points2[j], point1, point2, 10, cv::Scalar(100,100,100), 1); //key 1の向き
            draw_subline(*img, points2[j], point1, point2, -10, cv::Scalar(100,100,100), 1); //key 2の向き
            //ガイドライン5を書く
            cv::circle(*img, cv::Point(point3.x, point3.y), x_length/10, cv::Scalar(255, 0, 0), 1); //幅20の円を書く

            ///////////////////////////////ここまでok

        } else if (click == 3+n) {
            std::cout << "x_cm" <<std::to_string(x_cm) << std::endl;
            //＊step1
            point5 = tp(cv::Point(x,y));
            cv::circle(*img, point5, 4, cv::Scalar(0, 0, 255), -1);
            cv::putText(*img, "5", cv::Point(point5.x+5, point5.y-5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,255), 1.5);
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "5 (" << std::fixed << std::setprecision(2) << point5.x << ", " << point5.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*(n+n+2)+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
            std::cout << point_text.str() << std::endl;

            //ガイドライン5を消す
            cv::circle(*img, cv::Point(point3.x, point3.y), x_length/10, cv::Scalar(100, 100, 100), 1);


            //＊step2
            a4 = draw_line2(*img, point3, point5, cv::Scalar(100, 100, 255), 1);
            double theta = a1-a4;
            theta_text.str("");  // 文字列の内容をリセット
            theta_text.clear();  // 状態フラグをリセット
            theta_text << "theta " << std::fixed << std::setprecision(2) << (a1 - a4);
            cv::putText(*img, theta_text.str(), cv::Point(point3.x + 20, point3.y + 50), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,255,0), 1);
            cv::putText(*img, theta_text.str(), cv::Point(230, 50+(n+1)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            std::cout << a1 << std::endl;
            std::cout << a4 << std::endl;
            std::cout << "theta: " << a1-a4 << std::endl;

            //＊step3
            draw_subline(*img, point3, point3, points2[n], 20, cv::Scalar(0, 255, 0), 1); //key 1の向き
            draw_subline(*img, point3, point3, points2[n], -20, cv::Scalar(255, 0, 0), 1); //key 2の向き

            //＊step4
            update_display();
            int key = cv::waitKey(0);  // キー入力を待つ、ガイドラインの色を消す
            draw_subline(*img, point3, point3, points2[n], 20, cv::Scalar(100,100,100), 1); //key 1の向き
            draw_subline(*img, point3, point3, points2[n], -20, cv::Scalar(100,100,100), 1); //key 2の向き
            if (key == '1') {  // 1が押されたら、point3->point2_4に対して右側に線を引く
                a1 = draw_subline(*img, point3, point3, points2[n], x_length/10, cv::Scalar(100, 100, 255), 1);
            } else if (key == '2') {  // 2が押されたら、point3->point2_4に対して左側に線を引く
                a1 = draw_subline(*img, point3, point3, points2[n], -x_length/10, cv::Scalar(100, 100, 255), 1);
            }

            //＊step5
            draw_rectangle(*img, point2, point2, points2[n], points2[n], point3, cv::Scalar(150,150,150), 1);



            ////////////////////////////////////////////////////
            //＊step6

            double sum = 0.0;
            for (double value : phi_list){
                sum += value*value;
            }
            phi_rms = std::sqrt(sum/phi_list.size());
            phi_rms_text.str("");  // 文字列の内容をリセット
            phi_rms_text.clear();  // 状態フラグをリセット
            phi_rms_text << "phi_rms " << std::fixed << std::setprecision(2) << phi_rms;
            cv::putText(*img, phi_rms_text.str(), cv::Point(230, 50+(n+2)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            std::cout << "phi_rms: " << phi_rms << std::endl;

            double sum2 = 0.0;
            for (double value : phi_list){
                sum2 += (value*value - phi_rms*phi_rms)*(value*value - phi_rms*phi_rms);
            }
            phi_sigma = std::sqrt(sum2/phi_list.size());
            phi_sigma_text.str("");  // 文字列の内容をリセット
            phi_sigma_text.clear();  // 状態フラグをリセット
            phi_sigma_text << "phi_sigma " << std::fixed << std::setprecision(2) << phi_sigma;
            cv::putText(*img, phi_sigma_text.str(), cv::Point(230, 50+(n+3)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            std::cout << "phi_sigma: " << phi_sigma << std::endl;

            theta_0 = std::sqrt(3.0) * phi_rms;
            theta_0_text.str("");  // 文字列の内容をリセット
            theta_0_text.clear();  // 状態フラグをリセット
            theta_0_text << "theta_0 " << std::fixed << std::setprecision(2) << theta_0;
            cv::putText(*img, theta_0_text.str(), cv::Point(230, 50+(n+4)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            std::cout << "theta_0: " << theta_0 << std::endl;



            // ＊step7
            // theta_0とthetaとx_cmから、beta, beta gamma, (bpc)を求める
            // 今はelectronの場合を考えてる。muonにしたければmc2を変える

            double initial_guess = 0.5;
            double initial_guess_bpc = 1.0;
            double theta_rms = std::sqrt(theta*theta);
            beta_text.str("");  // 文字列の内容をリセット
            beta_text.clear();  // 状態フラグをリセット
            pb_ele_mc_text.str("");  // 文字列の内容をリセット
            pb_ele_mc_text.clear();  // 状態フラグをリセット
            bpc_text.str("");  // 文字列の内容をリセット
            bpc_text.clear();  // 状態フラグをリセット


            double mc2 = 0.511;


            std::function<double(double)> f1_fixed;
            f1_fixed = [&](double beta) { return f1(beta, x_cm, theta_rms * M_PI /180, mc2); };
            double beta_ans = fsolve(f1_fixed,initial_guess);
            beta_text << "beta " << std::fixed << std::setprecision(2) << beta_ans;
            cv::putText(*img, beta_text.str(), cv::Point(230, 50+(n+5)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            std::cout << "beta: " << beta_ans << std::endl;

            double pb_ele_mc = beta_ans / std::sqrt(1 - beta_ans * beta_ans);
            pb_ele_mc_text << "p/mc " << std::fixed << std::setprecision(2) << pb_ele_mc;
            cv::putText(*img, pb_ele_mc_text.str(), cv::Point(230, 50+(n+6)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            std::cout << "p/mc: " << pb_ele_mc << std::endl;

            // double bpc_ans = NewtonRaphson([&](double bpc) { return f2(bpc, x_cm, theta_rms*M_PI/180); },[&](double bpc) { return df2(bpc, x_cm); },initial_guess_bpc);
            // bpc_text << "bpc " << std::fixed << std::setprecision(2) << bpc_ans;
            // cv::putText(*img, bpc_text.str(), cv::Point(230, 50+(n+7)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            // std::cout << "bpc: " << bpc_ans << std::endl;


            beta_text.str("");  // 文字列の内容をリセット
            beta_text.clear();  // 状態フラグをリセット
            pb_ele_mc_text.str("");  // 文字列の内容をリセット
            pb_ele_mc_text.clear();  // 状態フラグをリセット
            bpc_text.str("");  // 文字列の内容をリセット
            bpc_text.clear();  // 状態フラグをリセット

            f1_fixed = [&](double beta) { return f1(beta, x_cm, theta_0 * M_PI /180, mc2); };
            beta_ans = fsolve(f1_fixed,initial_guess);
            beta_text << "beta_0 " << std::fixed << std::setprecision(2) << beta_ans;
            cv::putText(*img, beta_text.str(), cv::Point(230, 50+(n+8)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            std::cout << "beta_0: " << beta_ans << std::endl;

            pb_ele_mc = beta_ans / std::sqrt(1 - beta_ans * beta_ans);
            pb_ele_mc_text << "p/mc_0 " << std::fixed << std::setprecision(2) << pb_ele_mc;
            cv::putText(*img, pb_ele_mc_text.str(), cv::Point(230, 50+(n+9)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            std::cout << "p/mc_0: " << pb_ele_mc << std::endl;

            // bpc_ans = NewtonRaphson([&](double bpc) { return f2(bpc, x_cm, theta_0*M_PI/180); },[&](double bpc) { return df2(bpc, x_cm); },initial_guess_bpc);
            // bpc_text << "bpc_0 " << std::fixed << std::setprecision(2) << bpc_ans;
            // cv::putText(*img, bpc_text.str(), cv::Point(230, 50+(n+10)*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            // std::cout << "bpc_0: " << bpc_ans << std::endl;

            phi_list.clear();

            //////////////////////////////////////////////////ここまでたぶんok


        } else {
            click = 1;
            term++; 

            point1 = tp(cv::Point(x,y));
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "1 (" << std::fixed << std::setprecision(2) << point1.x << ", " << point1.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*0+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
            std::cout << point_text.str() << std::endl;

            cv::circle(*img, point1, 4, cv::Scalar(0, 0, 255), -1);
            cv::putText(*img, "1", cv::Point2f(point1.x + 5, point1.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,0,255), 1.5);
            draw_rectangle(*img, cv::Point(img->cols - 430, 20), cv::Point(0,0), cv::Point(300,0), cv::Point(0,0), cv::Point(0,40), cv::Scalar(255,0,0),2);
            cv::putText(*img, "select x_length", cv::Point(img->cols - 400, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);


            update_display();
            while (true) {
                int key = cv::waitKey(0);  // キー入力を待つ
                if (key == '1' || key == '2' || key == '3' || key == '4' || key == '5' || key == '6' || key == '7' || key == '8' || key == '9') {
                    set_x_length(key);
                    break;
                }
            }

            cv::putText(*img, "x  " + std::to_string(x_length), cv::Point(230, 50+0*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
            x_cm = x_length/cm;
            cv::putText(*img, "x cm " + std::to_string(x_cm), cv::Point(230, 50+1*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);



            //ガイドライン1を消す
            draw_rectangle(*img, cv::Point(img->cols - 430, 20), cv::Point(0,0), cv::Point(300,0), cv::Point(0,0), cv::Point(0,40), cv::Scalar(100,100,100),2);
            cv::putText(*img, "select x_length", cv::Point(img->cols - 400, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(100,100,100), 2);
            //ガイドライン2
            cv::circle(*img, point1, x_length/10, cv::Scalar(255, 0, 0), 1); //幅20の円を書く

            ////////////////////////ここまでたぶんok
        }
        update_display();
    }
}

//////////////////////////////////////////////////////////////////////////////////
int main() {

    std::ostringstream name_in, name_out;
    name_in << "./" << name << ".png";
    name_out << "./png_test/" << name << "_" << file <<".png";

    img = cv::imread(name_in.str()); 
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    cv::Mat img_bin;
    cv::cvtColor(img, img_bin, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_bin, img_color, cv::COLOR_GRAY2BGR);


    int max = x_9;
    cv::circle(img_color, cv::Point(img.cols-max-100, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "0", cv::Point(img.cols-max-100, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_1, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "1: 100", cv::Point(img.cols-max-100+x_1, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_2, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "2: 200", cv::Point(img.cols-max-100+x_2, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_3, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "3: 300", cv::Point(img.cols-max-100+x_3, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_4, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "4: 400", cv::Point(img.cols-max-100+x_4, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_5, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "5: 500", cv::Point(img.cols-max-100+x_5, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_6, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "6: 600", cv::Point(img.cols-max-100+x_6, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_7, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "7: 800", cv::Point(img.cols-max-100+x_7, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_8, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "8: 1000", cv::Point(img.cols-max-100+x_8, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::circle(img_color, cv::Point(img.cols-max-100+x_9, 100), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(img_color, "9: 1200", cv::Point(img.cols-max-100+x_9, 100 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.5);
    
    cv::line(img_color, cv::Point(img.cols-max-100, 100), cv::Point(img.cols - 100, 100), cv::Scalar(0, 0, 255), 2);
    cv::imshow("img_bin", img_color);

    cv::setMouseCallback("img_bin", at_click, &img_color);
    update_display();

    while (true) {
        int key = cv::waitKey(0);  // キー入力を待つ
        on_key(key);
    }

    return 0;
}