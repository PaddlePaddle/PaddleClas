//MIT License
//
//Copyright (c) 2021 Mobius Authors

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

//from https://github.com/sunbelbd/mobius/blob/e2d166547d61d791da8f06747a63b9cd38f02c71/main.cc

#include<stdio.h>
#include<string.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include<stdlib.h>
#include<memory>
#include<vector>
#include<functional>

#include"src/data.h"
#include"src/graph.h"

struct IndexContext{
    void* graph;
    void* data;
};


int topk = 0;
int display_topk = 1;
int build_idx_offset = 0;
int query_idx_offset = 0;

void flush_add_buffer(
        std::vector<std::pair<idx_t,std::vector<std::pair<int,value_t>>>>& add_buffer,
        GraphWrapper* graph){
    #pragma omp parallel for
    for(int i = 0;i < add_buffer.size();++i){
        auto& idx = add_buffer[i].first;
        auto& point = add_buffer[i].second;
        graph->add_vertex_lock(idx,point);
    }
    add_buffer.clear();
}


extern "C"{
// for mobius IP index
void build_mobius_index(float* dense_mat,int row,int dim, int pq_size, double mobius_pow , const char* prefix){
    std::unique_ptr<Data> data;
    std::unique_ptr<Data> data_original;
    std::unique_ptr<GraphWrapper> graph; 
    int topk = 0;
    int display_topk = 1;
    int build_idx_offset = 0;
    int query_idx_offset = 0;
        
    ++row;
    data = std::unique_ptr<Data>(new Data(row,dim));
    graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<3>(data.get())); 
    graph->set_construct_pq_size(pq_size);   
 
    std::vector<std::pair<idx_t,std::vector<std::pair<int,value_t>>>> add_buffer;
    
    ((FixedDegreeGraph<3>*)graph.get())->get_data()->mobius_pow = mobius_pow;
    data_original = std::unique_ptr<Data>(new Data(row,dim));

    std::vector<std::pair<int,value_t>> dummy_mobius_point;
    for(int i = 0;i < dim;++i)
        dummy_mobius_point.push_back(std::make_pair(i,0));
    
    //idx += build_idx_offset;
    
    for(int i = 0;i < row - 1;++i){

        std::vector<std::pair<int,value_t>> point;
        point.reserve(dim);
        for(int j = 0;j < dim;++j)
            point.push_back(std::make_pair(j,dense_mat[i * dim + j]));

        data_original->add(i,point);
        data->add_mobius(i,point);
        if(i < 1000){
            graph->add_vertex(i,point);
        }else{
            add_buffer.push_back(std::make_pair(i,point));
        }
        if(add_buffer.size() >= 1000000)
            flush_add_buffer(add_buffer,graph.get());
    }
    flush_add_buffer(add_buffer,graph.get());
    graph->add_vertex(row - 1,dummy_mobius_point);
    data.swap(data_original);

    std::string str = std::string(prefix);
    data->dump(str + ".data");
    graph->dump(str + ".graph");
 
}

void load_mobius_index_prefix(int row,int dim,IndexContext* index_context,const char* prefix){
    std::string str = std::string(prefix);
        
    ++row;
    Data* data = new Data(row,dim);
    GraphWrapper* graph = new FixedDegreeGraph<1>(data); 
    
    //idx += build_idx_offset;
    data->load(str + ".data");
    graph->load(str + ".graph");

    ((FixedDegreeGraph<1>*)graph)->search_start_point = row - 1;
    ((FixedDegreeGraph<1>*)graph)->ignore_startpoint = true;

    index_context->graph = graph;
    index_context->data = data;
}

void save_mobius_index_prefix(IndexContext* index_context,const char* prefix){
    std::string str = std::string(prefix);
    Data* data = (Data*)(index_context->data);
    GraphWrapper* graph = (GraphWrapper*)(index_context->graph); 

    data->dump(str + ".data");
    graph->dump(str + ".graph");
}

void search_mobius_index(float* dense_vec,int dim,int search_budget,int return_k, IndexContext* index_context,idx_t* ret_id,double* ret_score){
    int topk = 0;
    int display_topk = 1;
    int build_idx_offset = 0;
    int query_idx_offset = 0;
      
    Data* data = reinterpret_cast<Data*>(index_context->data);
    GraphWrapper* graph = reinterpret_cast<GraphWrapper*>(index_context->graph);
   
 
    //auto flag = (data==NULL);
    //std::cout<<flag<<std::endl;

    std::vector<std::pair<int,value_t>> point;
    point.reserve(dim);
    for(int j = 0;j < dim;++j)
        point.push_back(std::make_pair(j,dense_vec[j]));
    std::vector<idx_t> topN;
    std::vector<double> score;
    graph->search_top_k_with_score(point,search_budget,topN,score);
    for(int i = 0;i < topN.size() && i < return_k;++i){
        ret_id[i] = topN[i];
        ret_score[i] = score[i];
    }
}


// For L2 index
void build_l2_index(float* dense_mat,int row,int dim, int pq_size, const char* prefix){
    std::unique_ptr<Data> data;
    std::unique_ptr<GraphWrapper> graph; 
    int topk = 0;
    int display_topk = 1;
    int build_idx_offset = 0;
    int query_idx_offset = 0;
        
    data = std::unique_ptr<Data>(new Data(row,dim));
    graph = std::unique_ptr<GraphWrapper>(new FixedDegreeGraph<3>(data.get())); 
    graph->set_construct_pq_size(pq_size);

    std::vector<std::pair<idx_t,std::vector<std::pair<int,value_t>>>> add_buffer;
    
    for(int i = 0;i < row;++i){
        std::vector<std::pair<int,value_t>> point;
        point.reserve(dim);
        for(int j = 0;j < dim;++j)
            point.push_back(std::make_pair(j,dense_mat[i * dim + j]));
        data->add(i,point);
        if(i < 1000){
            graph->add_vertex(i,point);
        }else{
            add_buffer.push_back(std::make_pair(i,point));
        }
        if(add_buffer.size() >= 1000000)
            flush_add_buffer(add_buffer,graph.get());
    }
    flush_add_buffer(add_buffer,graph.get());
    
    std::string str = std::string(prefix);
    data->dump(str + ".data");
    graph->dump(str + ".graph");

}

void load_l2_index_prefix(int row,int dim,IndexContext* index_context,const char* prefix){
    std::string str = std::string(prefix);
        
    Data* data = new Data(row,dim);
    GraphWrapper* graph = new FixedDegreeGraph<3>(data); 
    
    //idx += build_idx_offset;

    data->load(str + ".data");
    graph->load(str + ".graph");

    index_context->graph = graph;
    index_context->data = data;
}

void save_l2_index_prefix(IndexContext* index_context,const char* prefix){
    std::string str = std::string(prefix);
    Data* data = (Data*)(index_context->data);
    GraphWrapper* graph = (GraphWrapper*)(index_context->graph); 

    data->dump(str + ".data");
    graph->dump(str + ".graph");
}



void search_l2_index(float* dense_vec,int dim,int search_budget,int return_k, IndexContext* index_context,idx_t* ret_id,double* ret_score){
    int topk = 0;
    int display_topk = 1;
    int build_idx_offset = 0;
    int query_idx_offset = 0;
        
    Data* data = reinterpret_cast<Data*>(index_context->data);
    GraphWrapper* graph = reinterpret_cast<GraphWrapper*>(index_context->graph);

    std::vector<std::pair<int,value_t>> point;
    point.reserve(dim);
    for(int j = 0;j < dim;++j)
        point.push_back(std::make_pair(j,dense_vec[j]));
    std::vector<idx_t> topN;
    std::vector<double> score;
    graph->search_top_k_with_score(point,search_budget,topN,score);
    for(int i = 0;i < topN.size() && i < return_k;++i){
//        printf("%d: (%zu, %f)\n",i,topN[i],score[i]);
        ret_id[i] = topN[i];
        ret_score[i] = score[i];
    }
}


void release_context(IndexContext* index_context){
    delete (Data*)(index_context->data);
    delete (GraphWrapper*)(index_context->graph);
}

} // extern "C"

