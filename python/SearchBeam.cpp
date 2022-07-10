#include <vector>
#include <map>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <omp.h>
#include "lm/state.hh"
#include "lm/virtual_interface.hh"
#include "lm/model.hh"
#include "SearchBeam.h"
#include "memviewslice.h"
using namespace std;

bool initialized=false;

template<class T, class Func>
T** create_and_init(int arr_length, Func func){
    T** res = new T*[arr_length];
    for(int i = 0; i < arr_length; i++){
        res[i] = func(i);
    }
    return res;
}

MultiThreadMemPool<SearchNode> sn_pool;
template<>
MultiThreadMemPool<SearchNode>::ThreadBuffer MultiThreadMemPool<SearchNode>::tbuf = MultiThreadMemPool<SearchNode>::ThreadBuffer();
MultiThreadMemPool<Notify> ntf_pool;
template<>
MultiThreadMemPool<Notify>::ThreadBuffer MultiThreadMemPool<Notify>::tbuf = MultiThreadMemPool<Notify>::ThreadBuffer();
#ifdef QUICKMAP_DEBUG
template<>
int QuickMap<int, float, SearchNode::QuickMapSize, dagstep_get_or_create>::access = 0;
template<>
int QuickMap<int, float, SearchNode::QuickMapSize, dagstep_get_or_create>::hard_access = 0;
#endif

NodeNotifyMap** node_notify_map_atomic;
int max_pos, max_batch_size;

lm::base::Model* model;

vector<pair<float, SearchNode*>>** beams;
MultiThreadMemPool<NodeStepMap::Node> ns_pool;
template<>
MultiThreadMemPool<NodeStepMap::Node>::ThreadBuffer MultiThreadMemPool<NodeStepMap::Node>::tbuf = MultiThreadMemPool<NodeStepMap::Node>::ThreadBuffer();
MultiThreadMemPool<NodeChildrenMap::Node> nc_pool;
template<>
MultiThreadMemPool<NodeChildrenMap::Node>::ThreadBuffer MultiThreadMemPool<NodeChildrenMap::Node>::tbuf = MultiThreadMemPool<NodeChildrenMap::Node>::ThreadBuffer();
MultiThreadMemPool<NodeNotifyMap::Node> nn_pool;
template<>
MultiThreadMemPool<NodeNotifyMap::Node>::ThreadBuffer MultiThreadMemPool<NodeNotifyMap::Node>::tbuf = MultiThreadMemPool<NodeNotifyMap::Node>::ThreadBuffer();
NodeStepMap** node_step_map;
NodeChildrenMap** node_children_map;



static NotifyCache thread_notify_cache;
static ExpandBeamCache thread_expand_cache;
# pragma omp threadprivate(thread_notify_cache, thread_expand_cache)


void global_init(int batch_size, int beam_size, int top_cand_n, int maxpos, int thread_num, char* lm_path)
{
    __printf("enter init\n");
    assert(!initialized);
    initialized = true;;
    __printf("create beam_size=%d top_cand_n=%d maxpos=%d thread_num=%d\n", beam_size, top_cand_n, maxpos, thread_num);
    
    max_batch_size = batch_size;
    int mempool_size = beam_size * top_cand_n * maxpos * batch_size + MultiThreadMemPool<SearchNode*>::buf_per_thread * thread_num * 2;
    sn_pool.init_global(mempool_size);
    ntf_pool.init_global(mempool_size);
    ns_pool.init_global(mempool_size);
    nc_pool.init_global(mempool_size);
    nn_pool.init_global(mempool_size);
    printf("dagsearch initialized: mempool_size=%d\n", mempool_size);
    
    max_pos = maxpos;

    beams = create_and_init<vector<pair<float, SearchNode*>>>(batch_size * maxpos, [&](int i){
        return new vector<pair<float, SearchNode*>>;
    });
    int hashsize = beam_size * top_cand_n * maxpos / 10;
    node_notify_map_atomic = create_and_init<NodeNotifyMap>(batch_size, [&](int i){
        return new NodeNotifyMap(hashsize, &nn_pool);
    });
    node_step_map = create_and_init<NodeStepMap>(batch_size, [&](int i){
        return new NodeStepMap(hashsize, &ns_pool);
    });
    node_children_map = create_and_init<NodeChildrenMap>(batch_size, [&](int i){
        return new NodeChildrenMap(hashsize, &nc_pool);
    });

    if(lm_path != nullptr){
        printf("loading lm\n");
        model = lm::ngram::LoadVirtual(lm_path, lm::ngram::Config());
        if(model != nullptr){
            printf("lm loading successfully\n");
        }else{
            printf("lm loading failed\n");
        }
    }else{
        model = nullptr;
    }

    #pragma omp parallel
    {
        thread_notify_cache.init();
        thread_expand_cache.init();
    }

    __printf("exit_init\n");
}
int query_vocab_index(char* word){
    const lm::base::Vocabulary &vocab = model->BaseVocabulary();
    return vocab.Index(word);
}

inline SearchNode* allocate_node(SearchNode* parent, int word, int lm_word)  // may be called parallelly
{
    SearchNode* now = sn_pool.allocate();
    #ifdef DEBUG
    if (now - sn_pool.pool >= ntf_pool.pool_size) printf("node memory exceeded!!!!!\n\n");
    #endif
    now->parent = parent;
    now->word = word;
    now->dagscore = -INFINITY;
    now->dagstepscore_map.clear();

    if(parent == nullptr){
        now->length = 0;
        if(model) model->BeginSentenceWrite(&now->lm_state);
        now->lmscore = 0;
    }else{
        now->length = parent->length + 1;
        if(model) now->lmscore = parent->lmscore + model->BaseScore(&parent->lm_state, lm_word, &now->lm_state);
        else now->lmscore = 0;
    }
    return now;
}

inline void insert_notify(int batch, SearchNode* target, int pos, int length)  // may be called parallelly
{
    Notify* now = ntf_pool.allocate();
    #ifdef DEBUG
    if (now - ntf_pool.pool >= ntf_pool.pool_size) printf("notify memory exceeded!!!!!\n\n");
    #endif
    now->target = target;
    auto& tar = (*thread_notify_cache.local_head)[make_pair(batch, make_pair(pos, length))];
    now->next = tar.first;
    tar.first = now;
    if(tar.second == nullptr) tar.second = now;
}

inline void direct_insert_notify(int batch, SearchNode* target, int pos, int length)
{
    Notify* now = ntf_pool.allocate();
        #ifdef DEBUG
    if (now - ntf_pool.pool >= ntf_pool.pool_size) printf("notify memory exceeded!!!!!\n\n");
    #endif
    now->target = target;
    bool create;
    now->next = node_notify_map_atomic[batch]->get_or_create(make_pair(pos, length), create, memory_order_relaxed).
                                               exchange(now, memory_order_relaxed);  //TODO: heat point
                                
    if(create) now->next = nullptr;
}

inline void add_step_dagscore(int batch, SearchNode* nextnode, int nextstep, float dagscore){
    bool create;
    // __printf("add_step_dagscore enter\n");
    float& target_dagscore = nextnode->dagstepscore_map.get_or_create(nextstep, create, batch, nextnode);
    if(create){ //write to notify if it's a new node for nextstep
        insert_notify(batch, nextnode, nextstep, nextnode->length);
        target_dagscore = dagscore;
    }else{
        target_dagscore = logaddexp(target_dagscore, dagscore);
    }
    // __printf("add_step_dagscore exit\n");
}

void init_start_node(int batch, int go_id)
{
    // __printf("init_start_node batch_id=%d\n", batch);
    SearchNode* node = allocate_node(nullptr, go_id, 0);
    // __printf("init_start_node after allocate node\n", batch);
    node->dagscore = 0;
    direct_insert_notify(batch, node, 0, 0);
    // __printf("init_start_node after notify\n", batch);
    bool create;
    float &dagscore = node->dagstepscore_map.get_or_create(0, create, batch, node);
    dagscore = 0;
    // __printf("init_start_node after insert node_step_map batch=%d\n", batch);
}

void init_beam(int batch_size, int go_id)
{
    assert(batch_size <= max_batch_size);
    #ifdef QUICKMAP_DEBUG
    QuickMap<int, float, SearchNode::QuickMapSize, dagstep_get_or_create>::show();
    #endif
    sn_pool.clear_global();
    ntf_pool.clear_global();
    ns_pool.clear_global();
    nc_pool.clear_global();
    nn_pool.clear_global();
    #pragma omp parallel
    {
        sn_pool.clear_thread();
        ntf_pool.clear_thread();
        ns_pool.clear_thread();
        nc_pool.clear_thread();
        nn_pool.clear_thread();
        #pragma omp for schedule(static) nowait
        for(int batch = 0; batch < batch_size; batch++){
            node_step_map[batch]->clear(); //hot
            node_children_map[batch]->clear(); //hot
            node_notify_map_atomic[batch]->clear();
            init_start_node(batch, go_id);
        }
    }
}



SearchNode* ExpandBeamCache::load(int batch, SearchNode* node, int nextword, int lm_word)
{
    if (node == search_node && nextword == search_nextword) return cached_nextnode;
    write_back();
    search_node = node; search_nextword = nextword;
    bool create;
    // __printf("cache load before query hash\n");
    SearchNode* &new_node = node_children_map[batch]->
            get_or_create(make_pair(node, nextword), create, memory_order_relaxed);
    // __printf("cache load after query hash\n");
    if(create) new_node = allocate_node(node, nextword, lm_word);
    cached_nextnode = new_node;
    cached_add_score = -INFINITY;
    // __printf("cache load cached_nextnode=%p\n", cached_nextnode);
    return cached_nextnode;
}

void ExpandBeamCache::write_back()
{
    if(cached_nextnode == nullptr) return;
    // __printf("write_back cached_nextnode=%p\n", cached_nextnode);
    float dagscore = cached_nextnode->dagscore;
    cached_nextnode->dagscore = logaddexp(dagscore, cached_add_score);
    cached_nextnode = nullptr;
    search_node = nullptr;
}

void NotifyCache::write_back()
{
    bool create;
    // printf("localhead: %p\n", local_head);
    for(auto &item : (*local_head)){
        int batch = item.first.first;
        Notify* now_head = item.second.first;
        Notify* now_tail = item.second.second;
        now_tail->next = node_notify_map_atomic[batch]->get_or_create(item.first.second, create, memory_order_relaxed).
                                                exchange(now_head, memory_order_relaxed);  //TODO: heat point
        if(create) now_tail->next = nullptr;
    }
    local_head->clear();
}

class ChunkManager{
public:
    vector<int> chk_arr;

    int prepare_chunk(int batch_size, int step, __Pyx_memviewslice output_length){
        int sum = 0;
        chk_arr.reserve(batch_size);
        for(int i = 0; i < batch_size; i++){
            chk_arr.push_back(sum);
            if(step < (*((int*)(output_length.data) + i) - 1)) sum += beams[i * max_pos]->size();
        }
        #ifdef DEBUG
            printf("prepare_chunk: sum=%d\n", sum);
        #endif
        return sum;
    }
        
    tuple<int, int> get(int i) const {
        int pos = upper_bound(chk_arr.begin(), chk_arr.end(), i) - chk_arr.begin() - 1;
        return {pos, i - chk_arr[pos]};
    }
};

inline void expand_path(int batch, SearchNode* node, int nextstep, int word, int lm_word, float dagscore)
{
    #ifdef DEBUG
    printf("expand_path batch=%d now=[", batch);
    __debug_print_node(node);
    printf("] nextstep=%d nextword=%d dagscore=%f\n", nextstep, word, dagscore);
    #endif

    SearchNode* nextnode_readonly = thread_expand_cache.load(batch, node, word, lm_word);
    thread_expand_cache.addscore(dagscore);
    add_step_dagscore(batch, nextnode_readonly, nextstep, dagscore);
}

template<>
void expand_beam(int batch_size, int step, 
            __Pyx_memviewslice output_length, 
            __Pyx_memviewslice dagscores, 
            __Pyx_memviewslice nextstep_idx, 
            __Pyx_memviewslice logits_idx, 
            __Pyx_memviewslice lm_vocab,
            float top_p) {

    int top_cand_n = dagscores.shape[2];

    ChunkManager chunk_manager;
    int chunk_size = chunk_manager.prepare_chunk(batch_size, step, output_length);

    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for(int i = 0; i < chunk_size; i++){
            int now_batch, now_beam;
            std::tie(now_batch, now_beam) = chunk_manager.get(i);

            // tid = omp_get_thread_num();
            // printf("expand_beam prange start tid=%d chunk=%d now_batch=%d now_beam=%d\n", tid, i, now_batch, now_beam);

            SearchNode* now_node = (*beams[now_batch * max_pos])[now_beam].second;

            bool create = false;
            float dagstepscore = now_node->dagstepscore_map.get_or_create(step, create, now_batch, now_node);

            #ifdef DEBUG
            if(create) printf("????????????? bug in expand_beam\n");
            #endif
            
            float count_sum = 0;
            for(int j = 0; j < top_cand_n; j++){
                if(count_sum < top_p){
                    int word = *((int*)(logits_idx.data + now_batch * logits_idx.strides[0] + step * logits_idx.strides[1]) + j);
                    int lm_word = *((int*)(lm_vocab.data) + word);
                    int nextstep = *((int*)(nextstep_idx.data + now_batch * nextstep_idx.strides[0] + step * nextstep_idx.strides[1]) + j);
                    float add_dagstepscore = *((float*)(dagscores.data + now_batch * dagscores.strides[0] + step * dagscores.strides[1]) + j);
                    count_sum += exp(add_dagstepscore);
                    expand_path(now_batch, now_node, nextstep, word, lm_word, dagstepscore + add_dagstepscore);
                }
            }

            //printf("expand_beam prange end tid=%d chunk=%d now_batch=%d now_beam=%d\n", tid, i, now_batch, now_beam);
        }

        thread_expand_cache.write_back();
        thread_notify_cache.write_back();
    }
}


