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

int click = 0;
int term = 0;
int enterkey = 1;
cv::Point2f point, point1, point2, point3, point4, point5;
double a0, a1, a2, a3, a4; 
double phi;
std::vector<double> phi_list;
double phi_rms, phi_sigma, theta_0;
std::string coo;
std::ostringstream phi_text, theta_text, phi_rms_text, phi_sigma_text, theta_0_text;
std::ostringstream beta_text, pb_ele_mc_text, bpc_text;
std::ostringstream point_text, text_line3;
double cm = 53.69; //ピクセル　霧箱変えたら変える必要あり


////////////////////////////////////////////////////////////////////////////////////
//測定の度に変える必要がありそうなところ
int N = 1; //何回目の測定か、Nに対応
int n = 8; //phiの測定数はn-1
std::string name = "example2"; //読み込むファイルの名前
std::string file = "click"; //書き出すファイルの名前に付け加えるところ
std::string namein_place = "./";
std::string nameout_place = "./test3_click_png/";
std::string namecsv_place = "./test3_click_csv/";

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


int m = 15; //文字の列の高さ指定
int height = (n+12)*m; //(n+n+2) > (n+10)だったら、(n+n+4)*mにする
std::vector<cv::Point> points2;
std::vector<cv::Point> result;



//////////////////////////////////////////////////////////////////////////

//ちょっと名前とか変えた
///////////////////////////////////////////////////////////////////////////////
//拡大・縮小、移動用
cv::Mat img, img_display, img_color;
//変数は上の//内に移動した

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

//全部一旦point2とかで取っておいて、そこからさらにその時点でtpしたやつを作っておく。全部そっちで書き直す
//lineとかに入れるときにtransformするのがいい <- これだとずれる
//この方法なら、click間で動いても大丈夫。ガイドラインの選択までの間には動けない
//ガイドラインの選択とかも全部clickで分ければ、pointも維持できるし、1手戻るとかの操作もできそう??
void at_click(int event, int x, int y, int flags, void* param) {
    if (event == cv::EVENT_LBUTTONDOWN) {  // 左クリック時
        click++;
        cv::Mat* img = static_cast<cv::Mat*>(param);

        std::ostringstream name_csv;
        name_csv << namecsv_place << name << ".csv";

        if (enterkey == 1){
            cv::Scalar color(0,0,255);
            point = tp(cv::Point(x,y));
            cv::circle(*img, point, 2, color, -1); 
            cv::putText(*img, "1", cv::Point2f(point.x + 5, point.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "1 (" << std::fixed << std::setprecision(2) << point.x << ", " << point.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            std::cout << point_text.str() << std::endl;
            writecsv(name_csv.str(), point.x, click, 2*N-1);
            writecsv(name_csv.str(), point.y, click, 2*N);
        } else if (enterkey == 2){
            cv::Scalar color(0,255,255);
            point = tp(cv::Point(x,y));
            cv::circle(*img, point, 2, color, -1); 
            cv::putText(*img, "2", cv::Point2f(point.x + 5, point.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << "2 (" << std::fixed << std::setprecision(2) << point.x << ", " << point.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            std::cout << point_text.str() << std::endl;
            writecsv(name_csv.str(), point.x, click, 2*N-1);
            writecsv(name_csv.str(), point.y, click, 2*N);
        } else if (enterkey == 3){
            cv::Scalar color(255,255,0);
            point = tp(cv::Point(x,y));
            cv::circle(*img, point, 2, color, -1); 
            cv::putText(*img, std::to_string(click), cv::Point2f(point.x + 5, point.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << std::to_string(click) << "(" << std::fixed << std::setprecision(2) << point.x << ", " << point.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            std::cout << point_text.str() << std::endl;
            writecsv(name_csv.str(), point.x, click, 2*N-1);
            writecsv(name_csv.str(), point.y, click, 2*N);
        } else if (enterkey ==4 ){
            cv::Scalar color(0,255,0);
            point = tp(cv::Point(x,y));
            cv::circle(*img, point, 2, color, -1); 
            cv::putText(*img, std::to_string(click), cv::Point2f(point.x + 5, point.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << std::to_string(click) << "(" << std::fixed << std::setprecision(2) << point.x << ", " << point.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            std::cout << point_text.str() << std::endl;
            writecsv(name_csv.str(), point.x, click, 2*N-1);
            writecsv(name_csv.str(), point.y, click, 2*N);
        } else if (enterkey == 5){
            cv::Scalar color(255,0,255);
            point = tp(cv::Point(x,y));
            cv::circle(*img, point, 2, color, -1); 
            cv::putText(*img, std::to_string(click), cv::Point2f(point.x + 5, point.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1.5);
            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << std::to_string(click) << "(" << std::fixed << std::setprecision(2) << point.x << ", " << point.y << ")";
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*click+term*height), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1.5);
            std::cout << point_text.str() << std::endl;
            writecsv(name_csv.str(), point.x, click, 2*N-1);
            writecsv(name_csv.str(), point.y, click, 2*N);


            point_text.str("");  // 文字列の内容をリセット
            point_text.clear();  // 状態フラグをリセット
            point_text << name << " n=" << std::to_string(click);
            cv::putText(*img, point_text.str(), cv::Point(20, 50+m*0+term*height), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(100,100,100), 2);
        } else {
            update_display(); //何もしない
        }
        update_display();
    }
}

//////////////////////////////////////////////////////////////////////////////////
int main() {

    std::ostringstream name_in, name_out;
    name_in << namein_place << name << ".png";
    name_out << nameout_place << name << "_" << std::to_string(N) << "_" << file <<".png";

    img = cv::imread(name_in.str()); 
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    cv::Mat img_bin;
    cv::cvtColor(img, img_bin, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_bin, img_color, cv::COLOR_GRAY2BGR);
    cv::imshow("img_bin", img_color);

    cv::setMouseCallback("img_bin", at_click, &img_color);
    update_display();

    while (true) {
        int key = cv::waitKey(0);  // キー入力を待つ
        on_key(key);
    }

    return 0;
}