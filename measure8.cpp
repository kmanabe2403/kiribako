#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <tuple>
#include <functional>
#include <vector>
#include <limits>
#include <string>

//画像保存がうまくいかないことがあったら、sを何回か押すようにすると大丈夫
////////////////////////////////////////////////////////////////////////////////////
//測定の度に変える必要がありそうなところ
int N = 5; //何回目の測定か
double x1 = 0.3; //cm 
int n = 200; //points2[i]の個数
double mc2 = 105.66; //muonかbetaかで変える
double initial_guess = 0.1; //muonなら0.1, betaなら0.5ぐらいがいいかも？

std::string name = "muon5"; //読み込むファイルの名前　beta2
std::string namecsv = "muon";
std::string file = "measure8"; //書き出すファイルの名前に付け加えるところ
std::string namein_place = "./png_lined/";
std::string nameout_place = "./measure8_png/";
std::string namecsv_place = "./measure8_csv/";


double cm = 53.69; //ピクセル　霧箱変えたら変える必要あり. scatter_20cm.cpp

//操作性微妙になったら、拡大率や移動量を変える
//拡大しすぎると移動がしにくい
//縮小するとき、一番左上にいかないと変になるかも
float scale = 1.0f;
cv::Point2f offset(0, 0);
const float scale_factor = 1.5f;  // 拡大率
const int move_step = 100;         // 移動量
const float min_scale = 1.0f;     // 最小拡大率
const float max_scale = 10.0f;     // 最大拡大率

//////////////////////////////////////////////////////////////////////////////////////


int click = 0;
int term = 0;
int enterkey = 1;
cv::Point2f point1, point2, point3, point30, point4, point5, point_pre;
double a2, a3; 
double phi;
std::vector<double> phi_list;
double phi_rms, phi_sigma, theta_0, beta_ans, pb_ele_mc;
std::ostringstream point_text, text_line3;



int m = 15; //文字の列の高さ指定
int height = (n+12)*m; //(n+n+2) > (n+10)だったら、(n+n+4)*mにする
std::vector<cv::Point2f> points2;
std::vector<cv::Point2f> points3, points30;
std::vector<cv::Point2f> result;

//////////////////////////////////////////////////////////////////////////////
//ある長さの線を引く(distance+length)　+n分割のところに点を打つ
std::tuple<std::vector<cv::Point2f>, double> draw_line_n(cv::Mat& img, cv::Point2f p1, cv::Point2f p2, int length, cv::Scalar color, int width) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    std::vector<cv::Point2f> points(n+1);
    points[0] = p2; //使わないけど、とりあえずiの数字合わせでいれておく

    if (distance > 0) {
        for (int i = 1; i<=n; i++) {
            double scale = 1 + length/distance * i;
            points[i] = cv::Point2f(p1.x + static_cast<int>(dx*scale), p1.y + static_cast<int>(dy*scale));
        }
        cv::line(img, p1, points[n], color, width);
    }
    double angle = std::atan2(-dy, dx);
    return std::make_tuple(points, angle * 180.0 / M_PI);
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


//angle[degree]のみ取り出す
double calculate_degree(cv::Point2f p1, cv::Point2f p2){
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    double angle = std::atan2(-dy, dx);
    return angle * 180.0 / M_PI;
}

//distanceのみ取り出す
double calculate_distance(cv::Point2f p1, cv::Point2f p2){
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double distance = std::sqrt(dx*dx + dy*dy);
    return distance;
}

//////////////////////////////////////////////////////////////////////////
// 数値微分（中央差分）
double numerical_derivative(std::function<double(double)> func, double x, double h = 1e-6) {
    return (func(x + h) - func(x - h)) / (2 * h);
}

// fsolve に近い非線形方程式の解法（数値微分付き Newton-Raphson + 二分法）
double fsolve(std::function<double(double)> func, double initialguess, double tol = 1e-6, int max_iter = 100) {
    double x = initialguess;
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

double f1(double beta, double x, double theta, double mc2) {
    const double z = 1.0;
    const double X0 = 30050.0;
    double log_term = std::log(std::max(x * z * z / (X0 * beta * beta), 1e-8));
    return 13.6 / mc2 * std::sqrt(std::max(1 - beta * beta, 1e-8)) / std::max(beta * beta, 1e-8) * z * std::sqrt(x / X0) * (1 + 0.038 * log_term) - theta;
}



///////////////////////////////////////////////////////////////////////////////
//拡大・縮小、移動用
cv::Mat img, img_display, img_color;
//変数は上の//内に移動した

//clickした画面上の点を画像上の点に変換する関数
cv::Point2f tp(cv::Point point) {
    return cv::Point2f((point.x + offset.x)/scale, (point.y + offset.y)/scale);
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
    name_out << nameout_place << name << "_" << std::to_string(N) << "_" << file <<".png";


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
        case 13:
            enterkey ++;
            break;
    }


    std::string display_text;
    if (enterkey == 1) {
        display_text = "now: point1";
        cv::rectangle(img_color, cv::Point(img_color.cols - 500, 110), cv::Point(img_color.cols - 50, 150), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img_color, display_text, cv::Point(img_color.cols-480, 130), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);
    } else if (enterkey == 2) {
        display_text = "now: point2";
        cv::rectangle(img_color, cv::Point(img_color.cols - 500, 110), cv::Point(img_color.cols - 50, 150), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img_color, display_text, cv::Point(img_color.cols-480, 130), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
    } else if (enterkey == 3) {
        display_text = "now: point3";
        cv::rectangle(img_color, cv::Point(img_color.cols - 500, 110), cv::Point(img_color.cols - 50, 150), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img_color, display_text, cv::Point(img_color.cols-480, 130), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 0), 2);
    } else if (enterkey == 5) {
        display_text = "now: point30";
        cv::rectangle(img_color, cv::Point(img_color.cols - 500, 110), cv::Point(img_color.cols - 50, 150), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img_color, display_text, cv::Point(img_color.cols-480, 130), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(100, 100, 100), 2);
    } else if (enterkey == 4) {
        display_text = "now: point4";
        cv::rectangle(img_color, cv::Point(img_color.cols - 500, 110), cv::Point(img_color.cols - 50, 150), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img_color, display_text, cv::Point(img_color.cols-480, 130), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    } else if (enterkey == 6) {
        display_text = "now: point5";
        cv::rectangle(img_color, cv::Point(img_color.cols - 500, 110), cv::Point(img_color.cols - 50, 150), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(img_color, display_text, cv::Point(img_color.cols-480, 130), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2);
    }
    update_display();
}


////////////////////////////////////////////////////////////////////////////////
//csvファイルへの書き込み用
void writecsv(const std::string& filename, double value, int row, int col) {
    std::ifstream file_in(filename);
    std::vector<std::vector<std::string>> data;
    std::string line;

    // ファイルが空でない場合に読み込み
    if (file_in) {
        // CSVファイルを読み込んでデータを格納
        while (std::getline(file_in, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row_data;
            while (std::getline(ss, cell, ',')) {
                row_data.push_back(cell);
            }
            data.push_back(row_data);
        }
    }
    file_in.close();

    // 行と列が範囲外なら、データを追加
    if (row - 1 >= data.size()) {
        data.resize(row);
    }

    // 指定された列が範囲外なら、その列を追加
    if (col - 1 >= data[row - 1].size()) {
        data[row - 1].resize(col);
    }

    // 3行目5列目に値を設定
    data[row - 1][col - 1] = std::to_string(value);

    // 変更したデータを再度CSVに書き込む
    std::ofstream file_out(filename);
    for (const auto& row_data : data) {
        for (size_t i = 0; i < row_data.size(); ++i) {
            file_out << row_data[i];
            if (i != row_data.size() - 1) {
                file_out << ",";  // 列の区切り
            }
        }
        file_out << "\n";  // 行の区切り
    }
    file_out.close();
    std::cout << "Value " << value << " written to row " << row << ", column " << col << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
double distance = 0;

void at_click(int event, int x, int y, int flags, void* param) {
    if (event == cv::EVENT_LBUTTONDOWN) {  // 左クリック時
        click++;
        cv::Mat* img = static_cast<cv::Mat*>(param);

        double x_cm;
        double x1px = x1*cm;

        std::ostringstream name_csv;
        name_csv << namecsv_place << namecsv << ".csv";
        
        if (enterkey == 1) {
            // BGR(0,0,255)
            // point1保存
            // x2に応じてpoint2を選ぶガイドライン引く
            cv::Scalar color(0,0,255);
            point1 = tp(cv::Point(x,y));
            cv::circle(*img, point1, 2, color, -1);
            cv::putText(*img, "1", cv::Point2f(point1.x + 5, point1.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);
            cv::circle(*img, point1, x1px, cv::Scalar(255,100,100), 1); //ガイドライン、半径x2の円を書く


            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "1 (" << std::fixed << std::setprecision(2) << point1.x << ", " << point1.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            writecsv(name_csv.str(), point1.x, click+10, 2*N-1);
            writecsv(name_csv.str(), point1.y, click+10, 2*N);

            point_pre = point1;


            


        } else if (enterkey == 2) {
            // BGR(0,255,255)
            // point2保存
            // point1からpoint2方向に、x2ごとにガイドラインを引く
            // points2[i]保存
            // points3[0]に、point2を代入
            // 方向選んで、垂直なガイドラインを引く
            //＊step1

            cv::Scalar color(0,255,255);
            point2 = tp(cv::Point(x,y));
            cv::circle(*img, point2, 2, color, -1);
            cv::putText(*img, "2", cv::Point2f(point2.x + 5, point2.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);

            auto [result, angle] = draw_line_n(*img, point1, point2, x1px, cv::Scalar(255, 100, 100), 1);
            points2 = result;
            for (int i = 1; i<=n; i++){
                //cv::circle(*img, points2[i], 2, cv::Scalar(255,100,100), -1);
                //cv::putText(*img, "2_"+std::to_string(i), cv::Point(points2[i].x + 5, points2[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(100,100,100), 1.5);
            }

            points3.push_back(point2);

            double subline_length = 1000; //px
            draw_subline(*img, point2, point1, point2, 20, cv::Scalar(0, 255, 0), 1); //key 1の向き　いる
            draw_subline(*img, point2, point1, point2, -20, cv::Scalar(255, 0, 0), 1); //key 2の向き　いる

            update_display();
            int key = cv::waitKey(0);  // キー入力を待つ、ガイドラインの色を消す
            draw_subline(*img, point2, point1, point2, 20, cv::Scalar(100,100,100), 1); //key 1の向き　いる
            draw_subline(*img, point2, point1, point2, -20, cv::Scalar(100,100,100), 1); //key 2の向き　いる
            if (key == '1') {  // 1が押されたら、point1->point2に対して右側に線を引く
                for (int i=1; i<=n; i++){
                    draw_subline(*img, points2[i], point1, point2, subline_length, cv::Scalar(255, 100, 100), 1); //いる
                    draw_subline(*img, points2[i], point1, point2, -subline_length/10, cv::Scalar(255, 100, 100), 1); //いる
                }
            } else if (key == '2') {  // 2が押されたら、point1->point2に対して左側に線を引く
                for (int i=1; i<=n; i++){
                    draw_subline(*img, points2[i], point1, point2, -subline_length, cv::Scalar(255, 100, 100), 1); //いる
                    draw_subline(*img, points2[i], point1, point2, subline_length/10, cv::Scalar(255, 100, 100), 1); //いる
                }
            }

            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "2 (" << std::fixed << std::setprecision(2) << point2.x << ", " << point2.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            writecsv(name_csv.str(), point2.x, click+10, 2*N-1);
            writecsv(name_csv.str(), point2.y, click+10, 2*N);


            distance += calculate_distance(point2, point_pre);
            point_pre = point2;



        } else if (enterkey == 3) {
            // BGR(255,255,0)
            // points3[i]保存
            cv::Scalar color(255,255,0);
            point3 = tp(cv::Point(x,y));
            cv::circle(*img, point3, 2, color, -1);
            //cv::putText(*img, "3_" + std::to_string(click-2), cv::Point2f(point3.x + 5, point3.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);
            points3.push_back(point3);

            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "3 (" << std::fixed << std::setprecision(2) << point3.x << ", " << point3.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            writecsv(name_csv.str(), point3.x, click+10, 2*N-1);
            writecsv(name_csv.str(), point3.y, click+10, 2*N);

            distance += calculate_distance(point3, point_pre);
            point_pre = point3;



        } else if (enterkey == 5) {
            // BGR(100,100,100)
            // points30[i]保存
            cv::Scalar color(255,255,255);
            point30 = tp(cv::Point(x,y));
            cv::circle(*img, point30, 2, color, -1);
            //cv::putText(*img, "3_" + std::to_string(click-2), cv::Point2f(point3.x + 5, point3.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);
            points30.push_back(point30);

            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "3' (" << std::fixed << std::setprecision(2) << point30.x << ", " << point30.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            writecsv(name_csv.str(), point30.x, click+10, 2*N-1);
            writecsv(name_csv.str(), point30.y, click+10, 2*N);

            distance += calculate_distance(point30, point_pre);
            point_pre = point30;


        } else if (enterkey == 4) {
            // BGR(0,255,0)
            // point4保存
            cv::Scalar color(0,255,0);
            point4 = tp(cv::Point(x,y));
            cv::circle(*img, point4, 2, color, -1);
            cv::putText(*img, "4", cv::Point2f(point4.x + 5, point4.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);

            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "4 (" << std::fixed << std::setprecision(2) << point4.x << ", " << point4.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            writecsv(name_csv.str(), point4.x, click+10, 2*N-1);
            writecsv(name_csv.str(), point4.y, click+10, 2*N);

            distance += calculate_distance(point4, point_pre);
            point_pre = point4;



        } else if (enterkey == 6) {
            // BGR(255,0,255)
            // point5保存
            cv::Scalar color(255,0,255);
            point5 = tp(cv::Point(x,y));
            cv::circle(*img, point5, 2, color, -1);
            cv::putText(*img, "5", cv::Point2f(point5.x + 5, point5.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);


            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "5 (" << std::fixed << std::setprecision(2) << point5.x << ", " << point5.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            writecsv(name_csv.str(), point5.x, click+10, 2*N-1);
            writecsv(name_csv.str(), point5.y, click+10, 2*N);

            distance += calculate_distance(point5, point_pre);

        
            for (int i=1; i<=points3.size()-1; i++){
                a2 = calculate_degree(points3[i], point4);
                a3 = calculate_degree(point2, points3[i]);
                phi = a2-a3;
                phi_list.push_back(phi);
                text_line3.str("");  // 文字列の内容をリセット
                text_line3.clear();  // 状態フラグをリセット
                text_line3 << "phi " << std::fixed << std::setprecision(2) << phi;
                cv::putText(*img, text_line3.str(), cv::Point(550, 50+i*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
            }

            double sum = 0.0;
            for (double value : phi_list){
                sum += value*value;
            }
            phi_rms = std::sqrt(sum/phi_list.size());

            double sum2 = 0.0;
            for (double value : phi_list){
                sum2 += (value*value - phi_rms*phi_rms)*(value*value - phi_rms*phi_rms);
            }
            phi_sigma = std::sqrt(sum2/phi_list.size());

            theta_0 = std::sqrt(3.0) * phi_rms;
            x_cm = x1 * (points3.size()-1);



            
            std::function<double(double)> f1_fixed;
            f1_fixed = [&](double beta) { return f1(beta, x_cm, theta_0 * M_PI /180, mc2); };
            beta_ans = fsolve(f1_fixed,initial_guess);
            pb_ele_mc = beta_ans / std::sqrt(1 - beta_ans * beta_ans);




            //1行目 click数
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "click: " << std::fixed << std::setprecision(0) << click;
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+1*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), static_cast<double>(click), 1, 2*N-1);
            //1行目 x2(x_cm)
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "x2: " << std::fixed << std::setprecision(2) << x_cm;
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+2*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), x_cm, 2, 2*N-1);
            //2行目 x1
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "x1: " << std::fixed << std::setprecision(2) << x1;
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+3*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), x1, 3, 2*N-1);
            //3行目 飛程(distance/cm)
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "range: " << std::fixed << std::setprecision(2) << distance/cm;
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+4*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), distance/cm, 4, 2*N-1);
            //4行目 phiの個数(phi_list.size())
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "numphi: " << std::fixed << std::setprecision(2) << phi_list.size();
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+5*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), static_cast<double>(phi_list.size()), 5, 2*N-1);
            //5行目 phi_rms
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "phi_rms: " << std::fixed << std::setprecision(2) << phi_rms;
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+6*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), phi_rms, 6, 2*N-1);
            //6行目 phi_sigma
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "phi_sigma: " << std::fixed << std::setprecision(2) << phi_sigma;
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+7*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), phi_sigma, 7, 2*N-1);
            //7行目 beta_0(beta_ans)
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "beta: " << std::fixed << std::setprecision(2) << beta_ans;
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+8*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), beta_ans, 8, 2*N-1);
            //8行目 p/mc(pb_ele_mc)
            text_line3.str("");  // 文字列の内容をリセット
            text_line3.clear();  // 状態フラグをリセット
            text_line3 << "p/mc: " << std::fixed << std::setprecision(2) << pb_ele_mc;
            cv::putText(*img, text_line3.str(), cv::Point(300, 50+9*m+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
            writecsv(name_csv.str(), pb_ele_mc, 9, 2*N-1);




        } else {
            click = 1;
            term++; 
            //何もしないことにした、飛跡を分割して測定したい場合は別のmeasureを使う
 
        }
        update_display();
    }
}

int main() {

    std::ostringstream name_in, name_out;
    name_in << namein_place << name << ".png";
    name_out << nameout_place << name << "_" << std::to_string(N) << "_" << file <<".png";

    img = cv::imread(name_in.str()); 
    cv::Mat img_bin;
    cv::cvtColor(img, img_bin, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_bin, img_color, cv::COLOR_GRAY2BGR);

    cv::rectangle(img_color, cv::Point(img_color.cols - 500, 20), cv::Point(img_color.cols - 50, 150), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(img_color, name_in.str(), cv::Point(img_color.cols-480, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);
    text_line3.str("");  // 文字列の内容をリセット
    text_line3.clear();  // 状態フラグをリセット
    text_line3 << "x1= " << std::fixed << std::setprecision(2) << x1 << " [cm]";
    cv::putText(img_color,text_line3.str() + "  N=" + std::to_string(N), cv::Point(img_color.cols-480, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);
    cv::putText(img_color, "now: point1", cv::Point(img_color.cols-480, 130), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);
    cv::imshow("img_bin", img_color);

    cv::setMouseCallback("img_bin", at_click, &img_color);
    update_display();

    while (true) {
        int key = cv::waitKey(0);  // キー入力を待つ
        on_key(key);
    }

    return 0;
}