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

//from https://github.com/sunbelbd/mobius/blob/e2d166547d61d791da8f06747a63b9cd38f02c71/graph.h


#pragma once
#include<vector>
#include<algorithm>
#include<queue>
#include<stdlib.h>
#include<random>
#include<unordered_set>
#include<mutex>
#include<time.h>

#include"config.h"
#include"data.h"

#ifdef OMP
#include<omp.h>
#endif

typedef unsigned int vl_type;

class VisitedList {
public:
	vl_type curV;
	vl_type *mass;
	unsigned int numelements;

	VisitedList(int numelements1) {
		curV = 1;
		numelements = numelements1;
		mass = new vl_type[numelements];
		memset(mass, 0, sizeof(vl_type) * numelements);
	}

	void reset() {
		++curV;
		if (curV == 0) {
			curV = 1;
			memset(mass, 0, sizeof(vl_type) * numelements);
		}
	};

	~VisitedList() { delete mass; }
};

struct GraphMeasures{
	int distance_cnt = 0;
};

class GraphWrapper{
public:
    virtual void add_vertex(idx_t vertex_id,std::vector<std::pair<int,value_t>>& point) = 0;
    virtual void add_vertex_lock(idx_t vertex_id,std::vector<std::pair<int,value_t>>& point) = 0;
    virtual void search_top_k(const std::vector<std::pair<int,value_t>>& query,int k,std::vector<idx_t>& result) = 0;
    virtual void search_top_k_with_score(const std::vector<std::pair<int,value_t>>& query,int k,std::vector<idx_t>& result,std::vector<double>& score){}

    virtual void dump(std::string path = "bfsg.graph") = 0;
    virtual void load(std::string path = "bfsg.graph") = 0;
    virtual ~GraphWrapper(){}
    virtual void set_construct_pq_size(int size){};
	GraphMeasures measures;
};

template<const int dist_type>
class FixedDegreeGraph : public GraphWrapper{
private:
    const int degree = SEARCH_DEGREE;
    const int flexible_degree = FIXED_DEGREE;
    const int vertex_offset_shift = FIXED_DEGREE_SHIFT;
    std::vector<idx_t> edges;
    std::vector<dist_t> edge_dist;
    Data* data;
    std::mt19937_64 rand_gen = std::mt19937_64(1234567);//std::random_device{}());
	std::vector<std::mutex> edge_mutex;//do not push back on this vector, it will destroy the mutex

	bool debug = false;
	VisitedList* p_visited = NULL;
	#ifdef OMP
	std::vector<VisitedList*> visited_pool;
	#endif
    int construct_pq_size = CONSTRUCT_SEARCH_BUDGET;


    void rank_and_switch_ordered(idx_t v_id,idx_t u_id){
        //We assume the neighbors of v_ids in edges[offset] are sorted 
        //by the distance to v_id ascendingly when it is full
        //NOTICE: before it is full, it is unsorted
        auto curr_dist = pair_distance(v_id,u_id);
        auto offset = ((size_t)v_id) << vertex_offset_shift;
		int degree = edges[offset];
		std::vector<idx_t> neighbor;
		neighbor.reserve(degree + 1);
		for(int i = 0;i < degree;++i)
			neighbor.push_back(edges[offset + i + 1]);
		neighbor.push_back(u_id);
		neighbor = edge_selection_filter_neighbor(neighbor,v_id,flexible_degree);
		edges[offset] = neighbor.size();
		for(int i = 0;i < neighbor.size();++i)
			edges[offset + i + 1] = neighbor[i];
		return;
        //We assert edges[offset] > 0 here
        if(curr_dist >= edge_dist[offset + edges[offset]]){
            return;
        }
        edges[offset + edges[offset]] = u_id;
        edge_dist[offset + edges[offset]] = curr_dist;
        for(size_t i = offset + edges[offset] - 1;i > offset;--i){
            if(edge_dist[i] > edge_dist[i + 1]){
                std::swap(edges[i],edges[i + 1]);
                std::swap(edge_dist[i],edge_dist[i + 1]);
            }else{
                break;
            }
        }
    }
    
    void rank_and_switch(idx_t v_id,idx_t u_id){
        rank_and_switch_ordered(v_id,u_id);
        //TODO:
        //Implement an unordered version to compare with
    }

    template<class T>
    dist_t distance(idx_t a,T& b){
		if(dist_type == 0)
        	       return data->l2_distance(a,b);
		else if(dist_type == 1)
        	       return data->negative_inner_prod_distance(a,b);
                else if(dist_type == 2)
		       return data->negative_cosine_distance(a,b);
		else if(dist_type == 3)
                        return  data->l2_distance(a,b);
		else if(dist_type == 4)
			return data->ipwrap_l2_build_distance(a,b);
		else if(dist_type == 5)
			return data->ipwrap_l2_query_distance(a,b);
                else{
			// should not happen
			fprintf(stderr,"unsupported dist_type %d\n",dist_type);
			return 0;
		}
    }

    void compute_distance_naive(size_t offset,std::vector<dist_t>& dists){
        dists.resize(edges[offset]);
        auto degree = edges[offset];
        for(int i = 0;i < degree;++i){
            dists[i] = distance(offset >> vertex_offset_shift,edges[offset + i + 1]);
        }
    }

    void compute_distance(size_t offset,std::vector<dist_t>& dists){
        compute_distance_naive(offset,dists);
    }
    
    template<class T>
    dist_t pair_distance_naive(idx_t a,T& b){
		++measures.distance_cnt;
        return distance(a,b);
    }

    template<class T>
    dist_t pair_distance(idx_t a,T& b){
        return pair_distance_naive(a,b);
    }
   

    void qsort(size_t l,size_t r){
        auto mid = (l + r) >> 1;
        int i = l,j = r;
        auto k = edge_dist[mid];
        do{
            while(edge_dist[i] < k) ++i;
            while(k < edge_dist[j]) --j;
            if(i <= j){
                std::swap(edge_dist[i],edge_dist[j]);
                std::swap(edges[i],edges[j]);
                ++i;
                --j;
            }
        }while(i <= j);
        if(i < r)qsort(i,r);
        if(l < j)qsort(l,j);
    }

    void rank_edges(size_t offset){
        std::vector<dist_t> dists;
        compute_distance(offset,dists);
        for(int i = 0;i < dists.size();++i)
            edge_dist[offset + i + 1] = dists[i];
        qsort(offset + 1,offset + dists.size());
        //TODO:
        //use a heap in the edge_dist
    }

    void add_edge_lock(idx_t v_id,idx_t u_id){
		edge_mutex[v_id].lock();
        auto offset = ((size_t)v_id) << vertex_offset_shift;
        if(edges[offset] < flexible_degree){
            ++edges[offset];
            edges[offset + edges[offset]] = u_id;
        }else{
            rank_and_switch(v_id,u_id);
        }
		edge_mutex[v_id].unlock();
    }

    void add_edge(idx_t v_id,idx_t u_id){
        auto offset = ((size_t)v_id) << vertex_offset_shift;
        if(edges[offset] < flexible_degree){
            ++edges[offset];
            edges[offset + edges[offset]] = u_id;
        }else{
            rank_and_switch(v_id,u_id);
        }
    }

public:
    long long total_explore_cnt = 0;
    int total_explore_times = 0;

	size_t search_start_point = 0;
	bool ignore_startpoint = false;

    FixedDegreeGraph(Data* data) : data(data){
        auto num_vertices = data->max_vertices();
        edges = std::vector<idx_t>(((size_t)num_vertices) << vertex_offset_shift);
        edge_dist = std::vector<dist_t>(((size_t)num_vertices) << vertex_offset_shift);
		edge_mutex = std::vector<std::mutex>(num_vertices);
		p_visited = new VisitedList(num_vertices + 5);
		#ifdef OMP
		int n_threads = 1;
		#pragma omp parallel
		#pragma omp master
		{
			n_threads = omp_get_num_threads();
		}
		visited_pool.resize(n_threads);
		for(int i = 0;i < n_threads;++i)
			visited_pool[i] = new VisitedList(num_vertices + 5);
		#endif
    }
    
    void set_construct_pq_size(int size){
        construct_pq_size = size; 
    }
   
	std::vector<idx_t> edge_selection_filter_neighbor(std::vector<idx_t>& neighbor,idx_t vertex_id,int desired_size){
		std::vector<idx_t> filtered_neighbor;
		std::vector<dist_t> dists(neighbor.size());
		for(int i = 0;i < dists.size();++i)
			dists[i] = pair_distance(vertex_id,neighbor[i]);
		std::vector<int> idx(neighbor.size());
		for(int i = 0;i < idx.size();++i)
			idx[i] = i;
		std::sort(idx.begin(),idx.end(),[&](int a,int b){return dists[a] < dists[b];});
		for(int i = 0;i < idx.size();++i){
			dist_t cur_dist = dists[idx[i]];
			bool pass = true;
			for(auto neighbor_id : filtered_neighbor){
				if(cur_dist > pair_distance(neighbor_id,neighbor[idx[i]])){
					pass = false;
					break;
				}
			}
			if(pass){
				filtered_neighbor.push_back(neighbor[idx[i]]);
				if(filtered_neighbor.size() >= desired_size)
					break;
			}else{
			}
		}
		return std::move(filtered_neighbor);
	}

    void add_vertex_lock(idx_t vertex_id,std::vector<std::pair<int,value_t>>& point){
        std::vector<idx_t> neighbor;
        search_top_k_lock(point,construct_pq_size,neighbor);
        auto offset = ((size_t)vertex_id) << vertex_offset_shift;
        int num_neighbors = degree < neighbor.size() ? degree : neighbor.size();
		edge_mutex[vertex_id].lock();
        // TODO:
        // it is possible to save this space --- edges[offset]
        // by set the last number in the range as 
        // a large number - current degree
		if(neighbor.size() >= degree)
			neighbor = edge_selection_filter_neighbor(neighbor,vertex_id,degree);
        edges[offset] = neighbor.size();

        for(int i = 0;i < neighbor.size() && i < degree;++i){
            edges[offset + i + 1] = neighbor[i]; 
        }
		edge_mutex[vertex_id].unlock();
        for(int i = 0;i < neighbor.size() && i < degree;++i){
            add_edge_lock(neighbor[i],vertex_id);
        }
    }
    void add_vertex(idx_t vertex_id,std::vector<std::pair<int,value_t>>& point){
        std::vector<idx_t> neighbor;
        search_top_k(point,construct_pq_size,neighbor);
        auto offset = ((size_t)vertex_id) << vertex_offset_shift;
        int num_neighbors = degree < neighbor.size() ? degree : neighbor.size();
        // TODO:
        // it is possible to save this space --- edges[offset]
        // by set the last number in the range as 
        // a large number - current degree
		if(neighbor.size() >= degree){
			neighbor = edge_selection_filter_neighbor(neighbor,vertex_id,degree);
		}
        edges[offset] = neighbor.size();

        for(int i = 0;i < neighbor.size() && i < degree;++i){
            edges[offset + i + 1] = neighbor[i]; 
        }
        for(int i = 0;i < neighbor.size() && i < degree;++i){
            add_edge(neighbor[i],vertex_id);
        }
    }
    
	void astar_multi_start_search_lock(const std::vector<std::pair<int,value_t>>& query,int k,std::vector<idx_t>& result){
        std::priority_queue<std::pair<dist_t,idx_t>,std::vector<std::pair<dist_t,idx_t>>,std::greater<std::pair<dist_t,idx_t>>> q;
        const int num_start_point = 1;

        auto converted_query = dist_type == 3 ? data->organize_point_mobius(query) : data->organize_point(query);
		#ifdef OMP
		int tid = omp_get_thread_num();
		auto& p_visited = visited_pool[tid];
		#endif

		p_visited->reset();
		auto tag = p_visited->curV;
        for(int i = 0;i < num_start_point && i < data->curr_vertices();++i){
            auto start = search_start_point;//rand_gen() % data->curr_vertices();
			if(p_visited->mass[start] == tag)
                continue;
			p_visited->mass[start] = tag;
            q.push(std::make_pair(pair_distance_naive(start,converted_query),start));
        }
        std::priority_queue<std::pair<dist_t,idx_t>> topk;
        const int max_step = 1000000;
        bool found_min_node = false;
        dist_t min_dist = 1e100;
        int explore_cnt = 0;
        for(int iter = 0;iter < max_step && !q.empty();++iter){
            auto now = q.top();
			if(topk.size() == k && topk.top().first < now.first){
                break;
            }
            ++explore_cnt;
            min_dist = std::min(min_dist,now.first);
            q.pop();
			if(ignore_startpoint == false || iter != 0)
	            topk.push(now);
            if(topk.size() > k)
                topk.pop();
			edge_mutex[now.second].lock();
            auto offset = ((size_t)now.second) << vertex_offset_shift;
            auto degree = edges[offset];

            for(int i = 0;i < degree;++i){
                auto start = edges[offset + i + 1];
			    if(p_visited->mass[start] == tag)
                    continue;
				p_visited->mass[start] = tag;
				auto dist = pair_distance_naive(start,converted_query);
				if(topk.empty() || dist < topk.top().first || topk.size() < k)
	                q.push(std::make_pair(dist,start));
            }
			edge_mutex[now.second].unlock();
        }
        total_explore_cnt += explore_cnt;
        ++total_explore_times;
        result.resize(topk.size());
        int i = result.size() - 1;
        while(!topk.empty()){
            result[i] = (topk.top().second);
            topk.pop();
            --i;
        }
    }
    
	void astar_no_heap_search(const std::vector<std::pair<int,value_t>>& query,std::vector<idx_t>& result){
        const int num_start_point = 1;
		std::pair<dist_t,idx_t> q_top = std::make_pair(10000000000,0);
        auto converted_query = dist_type == 3 ? data->organize_point_mobius(query) : data->organize_point(query);
		p_visited->reset();
		auto tag = p_visited->curV;
        for(int i = 0;i < num_start_point && i < data->curr_vertices();++i){
            auto start = search_start_point;//rand_gen() % data->curr_vertices();
			p_visited->mass[start] = tag;
			if(ignore_startpoint == false){
            	q_top = (std::make_pair(pair_distance_naive(start,converted_query),start));
			}else{
				auto offset = ((size_t)start) << vertex_offset_shift;
				auto degree = edges[offset];

				for(int i = 1;i <= degree;++i){
					p_visited->mass[edges[offset + i]] = tag;
					auto dis = pair_distance_naive(edges[offset + i],converted_query);
					if(dis < q_top.first)
						q_top = (std::make_pair(dis,start));
				}
			}
        }
        const int max_step = 1000000;
        bool found_min_node = false;
        dist_t min_dist = 1e100;
        int explore_cnt = 0;
        for(int iter = 0;iter < max_step;++iter){
            ++explore_cnt;
            auto offset = ((size_t)q_top.second) << vertex_offset_shift;
            auto degree = edges[offset];

			bool changed = false;
            for(int i = 0;i < degree;++i){
                auto start = edges[offset + i + 1];
			    if(p_visited->mass[start] == tag)
                    continue;
				p_visited->mass[start] = tag;
				auto dist = pair_distance_naive(start,converted_query);
				if(dist < q_top.first){
	                q_top = (std::make_pair(dist,start));
					changed = true;
				}
            }
			if(changed == false)
				break;
        }
        total_explore_cnt += explore_cnt;
        ++total_explore_times;
        result.resize(1);
		result[0] = q_top.second;
    }
    
    void astar_multi_start_search_with_score(const std::vector<std::pair<int,value_t>>& query,int k,std::vector<idx_t>& result,std::vector<double>& score){
        std::priority_queue<std::pair<dist_t,idx_t>,std::vector<std::pair<dist_t,idx_t>>,std::greater<std::pair<dist_t,idx_t>>> q;
        const int num_start_point = 1;

        auto converted_query = dist_type == 3 ? data->organize_point_mobius(query) : data->organize_point(query);
		p_visited->reset();
		auto tag = p_visited->curV;
        for(int i = 0;i < num_start_point && i < data->curr_vertices();++i){
            auto start = search_start_point;//rand_gen() % data->curr_vertices();
			if(p_visited->mass[start] == tag)
                continue;
			p_visited->mass[start] = tag;
            q.push(std::make_pair(pair_distance_naive(start,converted_query),start));
        }
        std::priority_queue<std::pair<dist_t,idx_t>> topk;
        const int max_step = 1000000;
        bool found_min_node = false;
        dist_t min_dist = 1e100;
        int explore_cnt = 0;
        for(int iter = 0;iter < max_step && !q.empty();++iter){
            auto now = q.top();
			if(topk.size() == k && topk.top().first < now.first){
                break;
            }
            ++explore_cnt;
            min_dist = std::min(min_dist,now.first);
            q.pop();
			if(ignore_startpoint == false || iter != 0)
	            topk.push(now);
            if(topk.size() > k)
                topk.pop();
            auto offset = ((size_t)now.second) << vertex_offset_shift;
            auto degree = edges[offset];

            for(int i = 0;i < degree;++i){
                auto start = edges[offset + i + 1];
			    if(p_visited->mass[start] == tag)
                    continue;
				p_visited->mass[start] = tag;
				auto dist = pair_distance_naive(start,converted_query);
				if(topk.empty() || dist < topk.top().first || topk.size() < k)
	                q.push(std::make_pair(dist,start));
            }
        }
        total_explore_cnt += explore_cnt;
        ++total_explore_times;
        result.resize(topk.size());
        score.resize(topk.size());
        int i = result.size() - 1;
        while(!topk.empty()){
            result[i] = (topk.top().second);
            score[i] = -(topk.top().first);
            topk.pop();
            --i;
        }
    }

    void astar_multi_start_search(const std::vector<std::pair<int,value_t>>& query,int k,std::vector<idx_t>& result){
        std::priority_queue<std::pair<dist_t,idx_t>,std::vector<std::pair<dist_t,idx_t>>,std::greater<std::pair<dist_t,idx_t>>> q;
        const int num_start_point = 1;

        auto converted_query = dist_type == 3 ? data->organize_point_mobius(query) : data->organize_point(query);
		p_visited->reset();
		auto tag = p_visited->curV;
        for(int i = 0;i < num_start_point && i < data->curr_vertices();++i){
            auto start = search_start_point;//rand_gen() % data->curr_vertices();
			if(p_visited->mass[start] == tag)
                continue;
			p_visited->mass[start] = tag;
            q.push(std::make_pair(pair_distance_naive(start,converted_query),start));
        }
        std::priority_queue<std::pair<dist_t,idx_t>> topk;
        const int max_step = 1000000;
        bool found_min_node = false;
        dist_t min_dist = 1e100;
        int explore_cnt = 0;
        for(int iter = 0;iter < max_step && !q.empty();++iter){
            auto now = q.top();
			if(topk.size() == k && topk.top().first < now.first){
                break;
            }
            ++explore_cnt;
            min_dist = std::min(min_dist,now.first);
            q.pop();
			if(ignore_startpoint == false || iter != 0)
	            topk.push(now);
            if(topk.size() > k)
                topk.pop();
            auto offset = ((size_t)now.second) << vertex_offset_shift;
            auto degree = edges[offset];

            for(int i = 0;i < degree;++i){
                auto start = edges[offset + i + 1];
			    if(p_visited->mass[start] == tag)
                    continue;
				p_visited->mass[start] = tag;
				auto dist = pair_distance_naive(start,converted_query);
				if(topk.empty() || dist < topk.top().first || topk.size() < k)
	                q.push(std::make_pair(dist,start));
            }
        }
        total_explore_cnt += explore_cnt;
        ++total_explore_times;
        result.resize(topk.size());
        int i = result.size() - 1;
        while(!topk.empty()){
            result[i] = (topk.top().second);
            topk.pop();
            --i;
        }
    }

    void search_top_k(const std::vector<std::pair<int,value_t>>& query,int k,std::vector<idx_t>& result){
		if(k == 1)
        	astar_no_heap_search(query,result);
		else
        	astar_multi_start_search(query,k,result);
    }
    
    void search_top_k_with_score(const std::vector<std::pair<int,value_t>>& query,int k,std::vector<idx_t>& result,std::vector<double>& score){
        astar_multi_start_search_with_score(query,k,result,score);
    }
    
	void search_top_k_lock(const std::vector<std::pair<int,value_t>>& query,int k,std::vector<idx_t>& result){
        astar_multi_start_search_lock(query,k,result);
    }

    void print_stat(){
        auto n = data->max_vertices();
        size_t sum = 0;
        std::vector<size_t> histogram(2 * degree + 1,0);
        for(size_t i = 0;i < n;++i){
            sum += edges[i << vertex_offset_shift];
            int tmp = edges[i << vertex_offset_shift];
            if(tmp > 2 * degree + 1)
                fprintf(stderr,"[ERROR] node %zu has %d degree\n",i,tmp);
            ++histogram[edges[i << vertex_offset_shift]];
            if(tmp != degree)
                fprintf(stderr,"[INFO] %zu has degree %d\n",i,tmp);
        }
        fprintf(stderr,"[INFO] #vertices %zu, avg degree %f\n",n,sum * 1.0 / n);
        std::unordered_set<idx_t> visited;
        fprintf(stderr,"[INFO] degree histogram:\n"); 
        for(int i = 0;i <= 2 * degree + 1;++i)
            fprintf(stderr,"[INFO] %d:\t%zu\n",i,histogram[i]);

    }
    
    void print_edges(int x){
        for(size_t i = 0;i < x;++i){
            size_t offset = i << vertex_offset_shift;
            int degree = edges[offset];
            fprintf(stderr,"%d (%d): ",i,degree);
            for(int j = 1;j <= degree;++j)
                fprintf(stderr,"(%zu,%f) ",edges[offset + j],edge_dist[offset + j]);
            fprintf(stderr,"\n");
        }
    }

    void dump(std::string path = "bfsg.graph"){
        FILE* fp = fopen(path.c_str(),"wb");
        size_t num_vertices = data->max_vertices();
        fwrite(&edges[0],sizeof(edges[0]) * (num_vertices << vertex_offset_shift),1,fp);
        fclose(fp);
    }

    void load(std::string path = "bfsg.graph"){
        FILE* fp = fopen(path.c_str(),"rb");
        size_t num_vertices = data->max_vertices();
        auto cnt = fread(&edges[0],sizeof(edges[0]) * (num_vertices << vertex_offset_shift),1,fp);
        fclose(fp);
    }

    Data* get_data(){
	return data;
    }

};

