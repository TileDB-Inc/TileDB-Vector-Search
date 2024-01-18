#ifndef DISKANN_GRAPH_HPP
#define DISKANN_GRAPH_HPP

#include <vector>

auto diskann_index_edge_list = std::vector<std::tuple<size_t, size_t>>{
    {0, 12},  {0, 14},  {0, 5},  {0, 9},   {1, 2},   {1, 12}, {1, 10}, {1, 4},
    {2, 1},   {2, 14},  {2, 9},  {3, 13},  {3, 6},   {3, 5},  {3, 11}, {4, 1},
    {4, 3},   {4, 7},   {4, 9},  {5, 3},   {5, 0},   {5, 8},  {5, 11}, {5, 13},
    {6, 3},   {6, 14},  {6, 7},  {6, 10},  {6, 13},  {7, 14}, {7, 4},  {7, 6},
    {8, 14},  {8, 5},   {8, 9},  {8, 12},  {9, 8},   {9, 4},  {9, 0},  {9, 2},
    {10, 14}, {10, 1},  {10, 9}, {10, 6},  {11, 3},  {11, 0}, {11, 5}, {12, 1},
    {12, 0},  {12, 8},  {12, 9}, {13, 3},  {13, 14}, {13, 5}, {13, 6}, {14, 7},
    {14, 2},  {14, 10}, {14, 8}, {14, 13},
};

std::vector<std::vector<size_t>> diskann_index_adj_list{
    {12, 14, 5, 9},
    {2, 12, 10, 4},
    {1, 14, 9},
    {13, 6, 5, 11},
    {1, 3, 7, 9},
    {3, 0, 8, 11, 13},
    {3, 14, 7, 10, 13},
    {14, 4, 6},
    {14, 5, 9, 12},
    {8, 4, 0, 2},
    {14, 1, 9, 6},
    {3, 0, 5},
    {1, 0, 8, 9},
    {3, 14, 5, 6},
    {7, 2, 10, 8, 13},
};

std::vector<std::tuple<size_t, float>> disk_ann_best{
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

// const TEST_DATA_FILE: &str = "tests/data/siftsmall_learn_256pts.fbin";
// const NUM_POINTS_TO_LOAD: usize = 256;

// let visited_nodes = index.search_for_point(&query, &mut scratch).unwrap();
// assert_eq!(visited_nodes.len(), 15);
// assert_eq!(scratch.best_candidates.size(), 15);

#endif  // DISKANN_GRAPH_HPP
