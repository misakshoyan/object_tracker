#pragma once

#include <vector>
#include "yolo_v2_class.hpp"

class Tracker
{
public:
    static Tracker& get_tracker()
    {
        static Tracker tracker;
        return tracker;
    }

public:
    Tracker() = default;
    Tracker(const Tracker&) = delete;
    Tracker(Tracker&&) = delete;
    Tracker& operator=(const Tracker&) = delete;
    Tracker& operator=(Tracker&&) = delete;
    ~Tracker() = default;

public:
    void find_matched_object_for_new_objects(const std::vector<bbox_t>& result_vec);
    void draw_boxes(cv::Mat& mat_img) const;

private:
    static std::string get_new_object_name() noexcept;

    bool ckeck_for_matching(const tracked_object& tr_obj, const bbox_t& new_obj, const ushort max_dist) const;
    void set_all_to_non_updated() noexcept;
    void set_non_updated_counts() noexcept;
    void remove_lost_objects() noexcept;

private:
    std::vector<tracked_object> m_tracked_objects;
    static constexpr short m_distance_for_tracking = 60;
};
