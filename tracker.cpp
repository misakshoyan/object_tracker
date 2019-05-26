#include "tracker.h"


std::string Tracker::get_new_object_name() noexcept
{
    // use integers as object names
    static unsigned int track_id = 0;
    return std::to_string(++track_id);
}

bool Tracker::ckeck_for_matching(const tracked_object& tr_obj, const bbox_t& new_obj, const ushort max_dist) const
{
    const cv::Point2d tr_obj_center(tr_obj.x + tr_obj.w / 2, tr_obj.y + tr_obj.h / 2);
    const cv::Point2d new_obj_center(new_obj.x + new_obj.w / 2, new_obj.y + new_obj.h / 2);

    const double x_diff = std::abs(tr_obj_center.x - new_obj_center.x);
    const double y_diff = std::abs(tr_obj_center.y - new_obj_center.y);

    if (sqrt(x_diff * x_diff + y_diff * y_diff) > max_dist) {
        return false;
    }

    return true;
}

void Tracker::set_all_to_non_updated() noexcept
{
    for (tracked_object& tr_obj : m_tracked_objects) {
        tr_obj.updated = false;
    }
}

void Tracker::set_non_updated_counts() noexcept
{
    for (tracked_object& tr_obj : m_tracked_objects) {
        if (!tr_obj.updated) {
            ++tr_obj.count_not_updated;
        }
    }
}

void Tracker::remove_lost_objects() noexcept
{
    m_tracked_objects.erase(std::remove_if(m_tracked_objects.begin(), m_tracked_objects.end(),
        [](const tracked_object& obj) { return obj.count_not_updated == 50; }),
        m_tracked_objects.end());
}

void Tracker::find_matched_object_for_new_objects(const std::vector<bbox_t>& result_vec)
{
    set_all_to_non_updated();

    for (const bbox_t& new_obj : result_vec) {
        bool found = false;
        for (tracked_object& tr_obj : m_tracked_objects) {
            if (ckeck_for_matching(tr_obj, new_obj, m_distance_for_tracking)) {
                found = true;
                tr_obj.x = new_obj.x;
                tr_obj.y = new_obj.y;
                tr_obj.w = new_obj.w;
                tr_obj.h = new_obj.h;
                tr_obj.updated = true;
                break;
            }
        }
        if (!found) {
            m_tracked_objects.emplace_back(new_obj.x, new_obj.y, new_obj.w, new_obj.h, get_new_object_name());
        }
    }

    set_non_updated_counts();
    remove_lost_objects();
}

void Tracker::draw_boxes(cv::Mat& mat_img) const
{
    for (auto &i : m_tracked_objects) {
        const cv::Scalar rect_color(30, 200, 30);
        const cv::Scalar text_color(0, 0, 0);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), rect_color, 3);
        putText(mat_img, i.obj_name, cv::Point2f(i.x, i.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, text_color);
    }
    cv::imshow("window name", mat_img);
    cv::waitKey(3);
}


