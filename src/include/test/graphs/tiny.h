#ifndef TINY_GRAPH_HPP
#define TINY_GRAPH_HPP

#include <vector>

auto tiny_vectors = std::vector<std::vector<size_t>>{
    {0, 0},  // 0
    {2, 0},  // 1
    {8, 0},  // 2
    {1, 1},  // 3
    {2, 2},  // 4
    {2, 3},  // 5
    {8, 2},  // 6
};

auto tiny_index_edge_list = std::vector<std::tuple<size_t, size_t>>{
    {0, 1},
    {1, 2},
    {1, 3},
    {2, 3},
    {2, 6},
    {3, 5},
    {3, 6},
    {4, 3},
    {5, 4},
};

std::vector<std::vector<size_t>> tiny_index_adj_list{
    {1},
    {2, 3},
    {3, 6},
    {5, 6},
    {3},
    {4},
};

std::vector<std::vector<std::tuple<float, size_t>>> tiny_adj_list{
    {{8.0, 1}},
    {{6.7, 2}, {5.0, 3}},
    {{1, 3}, {31, 6}},
    {{4, 5}, {15, 6}},
    {{9.9, 3}},
    {{99, 4}},
};

std::vector<std::tuple<size_t, float>> tiny_best{
    {2, 120899.0},
    {8, 145538.0},
    {72, 146046.0},
    {4, 148462.0},
    {7, 148912.0},
    {10, 154570.0},
    {1, 159448.0},
    {12, 170698.0},
    {9, 177205.0},
    {0, 259996.0},
    {6, 371819.0},
    {5, 385240.0},
    {3, 413899.0},
    {13, 416386.0},
    {11, 449266.0},
};

#endif  // DISKANN_GRAPH_HPP
