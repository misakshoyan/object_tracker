#include <iostream>
#include <string>
#include <vector>

// imported functions from DLL
#include "yolo_v2_class.hpp"

#include "tracker.h"


int main()
{
    cv::namedWindow("window name", cv::WINDOW_NORMAL);
    Tracker& tracker = Tracker::get_tracker();
    Detector detector("yolo-obj.cfg", "backup\\yolo-obj_1000.weights");

    std::cout << "enter video file path: ";
    std::string filename;
    std::cin >> filename;

    cv::Mat frame;

    for (cv::VideoCapture cap(filename); cap >> frame, cap.isOpened();) {
        std::vector<bbox_t> result_vec = detector.detect(frame, 0.2);
        tracker.find_matched_object_for_new_objects(result_vec);
        tracker.draw_boxes(frame);

        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}
