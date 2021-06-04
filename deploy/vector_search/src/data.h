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

//from https://github.com/sunbelbd/mobius/blob/e2d166547d61d791da8f06747a63b9cd38f02c71/data.h

#pragma once

#include<memory>
#include<vector>
#include<math.h>

#include"config.h"

#define ZERO_EPS 1e-10

#define _SCALE_WORLD_DENSE_DATA

#ifdef _SCALE_WORLD_DENSE_DATA
//dense data
class Data{
private:
    std::unique_ptr<value_t[]> data;
    size_t num;
    size_t curr_num = 0;
    int dim;

public:
	value_t mobius_pow = 2;
	value_t max_ip_norm = 1;
	value_t max_ip_norm2 = 1;

    Data(size_t num, int dim) : num(num),dim(dim){
        data = std::unique_ptr<value_t[]>(new value_t[num * dim]);
        memset(data.get(),0,sizeof(value_t) * num * dim);
    }
    
    value_t* get(idx_t idx) const{
        return data.get() + idx * dim;
    }
    
	template<class T>
    dist_t ipwrap_l2_query_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
		dist_t normu = 0;
        for(int i = 0;i < dim;++i){
            auto diff = (*(pa + i) / max_ip_norm) - v[i];
            ret += diff * diff;
			normu += (*(pa + i)) * (*(pa + i));
        }
		ret += 1 - normu / max_ip_norm2;
        return ret;
    }

    template<class T>
    dist_t ipwrap_l2_build_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
		dist_t normu = 0;
		dist_t normv = 0;
        for(int i = 0;i < dim;++i){
            auto diff = *(pa + i) - v[i];
            ret += diff * diff;
			normu += (*(pa + i)) * (*(pa + i));
			normv += v[i] * v[i];
        }
		dist_t wrap_termu = sqrt(1 - normu / max_ip_norm2);
		dist_t wrap_termv = sqrt(1 - normv / max_ip_norm2);
		dist_t diff_wrap = wrap_termu - wrap_termv;
		ret = ret / max_ip_norm2 + diff_wrap * diff_wrap;
        return ret;
    }

    template<class T>
    dist_t l2_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
        for(int i = 0;i < dim;++i){
            auto diff = *(pa + i) - v[i];
            ret += diff * diff;
        }
        return ret;
    }
    
    template<class T>
    dist_t negative_inner_prod_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
        for(int i = 0;i < dim;++i){
            ret -= (*(pa + i)) * v[i];
        }
        return ret;
    }
    
    template<class T>
    dist_t negative_cosine_distance(idx_t a,T& v) const{
        auto pa = get(a);
        dist_t ret = 0;
        value_t lena = 0,lenv = 0;
        for(int i = 0;i < dim;++i){
            ret += (*(pa + i)) * v[i];
            lena += (*(pa + i)) * (*(pa + i));
            lenv += v[i] * v[i];
        }
        int sign = ret < 0 ? 1 : -1;
//        return sign * (ret * ret / lena);// / lenv);
        return sign * (ret * ret / lena / lenv);
    }
    
	template<class T>
    dist_t mobius_l2_distance(idx_t a,T& v) const{
		auto pa = get(a);
		dist_t ret = 0;
		value_t lena = 0,lenv = 0;
		for(int i = 0;i < dim;++i){
			lena += (*(pa + i)) * (*(pa + i));
			lenv += v[i] * v[i];
		}
		value_t modifier_a = pow(lena,0.5 * mobius_pow);
		value_t modifier_v = pow(lenv,0.5 * mobius_pow);
		if(fabs(modifier_a) < ZERO_EPS)
			modifier_a = 1;
		if(fabs(modifier_v) < ZERO_EPS)
			modifier_v = 1;
		for(int i = 0;i < dim;++i){
			value_t tmp = (*(pa + i)) / modifier_a - v[i] / modifier_v;
			ret += tmp * tmp;
		}
		return ret;
    }

    template<class T>
    dist_t real_nn(T& v) const{
        dist_t minn = 1e100;
        for(size_t i = 0;i < curr_num;++i){
            auto res = l2_distance(i,v);
            if(res < minn){
                minn = res;
            }
        }
        return minn;
    }
    
    std::vector<value_t> organize_point_mobius(const std::vector<std::pair<int,value_t>>& v){
        std::vector<value_t> ret(dim,0);
		value_t lena = 0;
        for(const auto& p : v){
//            ret[p.first] = p.second;
			lena += p.second * p.second;
        }
		value_t modifier_a = pow(lena,0.5 * mobius_pow);
		if(fabs(modifier_a) < ZERO_EPS)
			modifier_a = 1;
        for(const auto& p : v){
            ret[p.first] = p.second / modifier_a;
        }
        return std::move(ret);
    }
    
    std::vector<value_t> organize_point(const std::vector<std::pair<int,value_t>>& v){
        std::vector<value_t> ret(dim,0);
        for(const auto& p : v){
            if(p.first >= dim)
                printf("error %d %d\n",p.first,dim);
            ret[p.first] = p.second;
        }
        return std::move(ret);
    }

    value_t vec_sum2(const std::vector<std::pair<int,value_t>>& v){
        value_t ret = 0;
        for(const auto& p : v){
            if(p.first >= dim)
                printf("error %d %d\n",p.first,dim);
            ret += p.second * p.second;
        }
        return std::move(ret);
    }


    void add(idx_t idx, std::vector<std::pair<int,value_t>>& value){
        //printf("adding %zu\n",idx);
        //for(auto p : value)
        //    printf("%zu %d %f\n",idx,p.first,p.second);
        curr_num = std::max(curr_num,idx);
        auto p = get(idx);
        for(const auto& v : value)
            *(p + v.first) = v.second;
    }

	void add_mobius(idx_t idx, std::vector<std::pair<int,value_t>>& value){
        //printf("adding %zu\n",idx);
        //for(auto p : value)
        //    printf("%zu %d %f\n",idx,p.first,p.second);
        curr_num = std::max(curr_num,idx);
        auto p = get(idx);
		value_t lena = 0;
        for(const auto& v : value){
            *(p + v.first) = v.second;
			lena += v.second * v.second;
		}
		value_t modifier_a = pow(lena,0.5 * mobius_pow);
		if(fabs(modifier_a) < ZERO_EPS)
			modifier_a = 1;
        for(const auto& v : value){
            *(p + v.first) = v.second / modifier_a;
		}
    }

    inline size_t max_vertices(){
        return num;
    }

    inline size_t curr_vertices(){
        return curr_num;
    }

    void print(){
        for(int i = 0;i < num && i < 10;++i)
            printf("%f ",*(data.get() + i));
        printf("\n");
    }

    int get_dim(){
        return dim;
    }

    void dump(std::string path = "bfsg.data"){
        FILE* fp = fopen(path.c_str(),"wb");
        fwrite(data.get(),sizeof(value_t) * num * dim,1,fp);
        fclose(fp);
    }
    
    void load(std::string path = "bfsg.data"){
        curr_num = num;
        FILE* fp = fopen(path.c_str(),"rb");
        auto cnt = fread(data.get(),sizeof(value_t) * num * dim,1,fp);
        fclose(fp);
    }

};
template<>
dist_t Data::ipwrap_l2_build_distance(idx_t a,idx_t& b) const{
	auto pa = get(a);
	auto pb = get(b);
	dist_t ret = 0;
	dist_t normu = 0;
	dist_t normv = 0;
	for(int i = 0;i < dim;++i){
        auto diff = *(pa + i) - *(pb + i);
		ret += diff * diff;
		normu += (*(pa + i)) * (*(pa + i));
		normv += (*(pb + i)) * (*(pb + i));
	}
	dist_t wrap_termu = sqrt(1 - normu / max_ip_norm2);
	dist_t wrap_termv = sqrt(1 - normv / max_ip_norm2);
	dist_t diff_wrap = wrap_termu - wrap_termv;
	ret = ret / max_ip_norm2 + diff_wrap * diff_wrap;
	return ret;
}
template<>
dist_t Data::ipwrap_l2_query_distance(idx_t a,idx_t& b) const{
	auto pa = get(a);
	auto pb = get(b);
	dist_t ret = 0;
	dist_t normu = 0;
	for(int i = 0;i < dim;++i){
        auto diff = (*(pa + i) / max_ip_norm) - *(pb + i);
		ret += diff * diff;
		normu += (*(pa + i)) * (*(pa + i));
	}
	ret += 1 - normu / max_ip_norm2;
	return ret;
}
template<>
dist_t Data::l2_distance(idx_t a,idx_t& b) const{
    auto pa = get(a),
         pb = get(b);
    dist_t ret = 0;
    for(int i = 0;i < dim;++i){
        auto diff = *(pa + i) - *(pb + i);
        ret += diff * diff;
    }
    return ret;
}

template<>
dist_t Data::negative_inner_prod_distance(idx_t a,idx_t& b) const{
    auto pa = get(a),
         pb = get(b);
    dist_t ret = 0;
    for(int i = 0;i < dim;++i){
        ret -= (*(pa + i)) * (*(pb + i));
    }
    return ret;
}

template<>
dist_t Data::negative_cosine_distance(idx_t a,idx_t& b) const{
    auto pa = get(a),
         pb = get(b);
    dist_t ret = 0;
    value_t lena = 0,lenv = 0;
    for(int i = 0;i < dim;++i){
        ret += (*(pa + i)) * (*(pb + i));
        lena += (*(pa + i)) * (*(pa + i));
        lenv += (*(pb + i)) * (*(pb + i));
    }
    int sign = ret < 0 ? 1 : -1;
//    return sign * (ret * ret / lena);
    return sign * (ret * ret / lena / lenv);
}

template<>
dist_t Data::mobius_l2_distance(idx_t a,idx_t& b) const{
    auto pa = get(a),
         pb = get(b);
    dist_t ret = 0;
    value_t lena = 0,lenv = 0;
    for(int i = 0;i < dim;++i){
        lena += (*(pa + i)) * (*(pa + i));
        lenv += (*(pb + i)) * (*(pb + i));
    }
	value_t modifier_a = pow(lena,0.5 * mobius_pow);
	value_t modifier_v = pow(lenv,0.5 * mobius_pow);
	if(fabs(modifier_a) < ZERO_EPS)
		modifier_a = 1;
	if(fabs(modifier_v) < ZERO_EPS)
		modifier_v = 1;
    for(int i = 0;i < dim;++i){
        value_t tmp = (*(pa + i)) / modifier_a - (*(pb + i)) / modifier_v;
		ret += tmp * tmp;
    }
    return ret;
}

#else
//sparse data
class Data{
public:
    //TODO

};
#endif


