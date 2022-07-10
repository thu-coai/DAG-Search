#include <vector>
#include <map>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cassert>
#include "lm/state.hh"
#include "lm/virtual_interface.hh"
#include "lm/model.hh"
using namespace std;
// #define DEBUG

template<class K, class V, int num, class fallback_get_or_create>
class QuickMap // Store several (key, value) pairs. If full, use fallback_get_or_create.
{
public:
    atomic<int> len_atomic;
    pair<K, V> kv[num];

#ifdef QUICKMAP_DEBUG
    static int access, hard_access;
#endif

    void clear(){
        len_atomic.store(0, memory_order_relaxed);
    }

#ifdef QUICKMAP_DEBUG
    static void show(){
        if(access != 0){
            printf("QuickMap access=%d hard_access=%d rate=%.2f\n", access, hard_access, (float)hard_access / access * 100);
            access = hard_access = 0;
        }
    }
#endif

    template<typename... Args>
    V& get_or_create(const K &k, bool &create, Args&&... args){
#ifdef QUICKMAP_DEBUG
        access ++;
#endif
        int len = len_atomic.load(memory_order_acquire); // only support one write and multiple reads with different key
        if(num > 0 && len <= num){
            create = false;
            for(int i = 0; i < num; i++){
                if(i < len && kv[i].first == k) return kv[i].second;
            }

            // create array
            if(len < num) {
                create = true;
                len++;
                kv[len-1].first = k;
                len_atomic.store(len, memory_order_release);
                return kv[len-1].second;
            }

            //create map
            for(int i = 0; i < num; i++){
                fallback_get_or_create()(kv[i].first, create, std::forward<Args>(args)...) = kv[i].second;
            }
            //fall through to map insert
        }
#ifdef QUICKMAP_DEBUG
        hard_access++;
#endif
        V& res = fallback_get_or_create()(k, create, std::forward<Args>(args)...);
        if(len == num) len_atomic.store(num + 1, memory_order_release);
        return res;
    }
};

struct SearchNode;
class dagstep_get_or_create{
public:
    inline float& operator()(int nextstep, bool &create, int batch_id, SearchNode* nextnode);
};


struct SearchNode
{
    SearchNode *parent; 
    int word, length;
    
    float lmscore, dagscore;
    static const int QuickMapSize = 5;
    QuickMap<int, float, QuickMapSize, dagstep_get_or_create> dagstepscore_map;
    lm::ngram::State lm_state;
};

#ifdef DEBUG
const bool __debug_flag = true;
#else
const bool __debug_flag = false;
#endif

inline int __printf (const char *format, ...)
{
   va_list arg;
   int done;

   va_start (arg, format);
   done = vfprintf (stdout, format, arg);
   va_end (arg);

   return done;
}
inline void __debug_print_node(SearchNode* now){
    if(now == nullptr) return;
    __debug_print_node(now->parent);
    __printf("%d ", now->word);
}

inline float logaddexp(float a, float b){
    float l = max(a, b);
    if(isinf(l) != 0) return -INFINITY;
    return log(exp(a - l) + exp(b - l)) + l;
}

inline bool node_compare_allscore(const pair<float, SearchNode*> &a, const pair<float, SearchNode*> & b){
    return a.first > b.first;
}
inline float calculate_score(SearchNode* node, float alpha, float gamma){
    return (node->lmscore * gamma + node->dagscore) / pow(node->length, alpha);
}

struct Notify
{
    SearchNode* target;
    Notify* next;
};

class ExpandBeamCache
{
public:
    SearchNode* cached_nextnode;
    float cached_add_score;
    SearchNode* search_node;
    int search_nextword;

    void init() { cached_nextnode = nullptr; search_node = nullptr; }
    SearchNode* load(int batch, SearchNode* node, int nextword, int lm_word);
    void write_back();
    void addscore(float dagscore) { cached_add_score = logaddexp(cached_add_score, dagscore); }
};

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return operator()(pair.first) ^ operator()(pair.second);
    }

    template <class T>
    std::size_t operator() (const T &ele) const {
        return std::hash<T>()(ele);
    }
};

class NotifyCache
{
public:
    unordered_map<pair<int, pair<int, int>>, pair<Notify*, Notify*>, pair_hash> *local_head;

    void init(){ 
        local_head = new unordered_map<pair<int, pair<int, int>>, pair<Notify*, Notify*>, pair_hash>;
    }

    void write_back();
};

template<class T>
class MultiThreadMemPool
{
public:
    T* pool;
    atomic<T*> shared_pool_pt;
    int pool_size;
    static const int buf_per_thread = 1024, randomized_buf_per_thread = 1024;

    struct ThreadBuffer
    {
        T *private_pool_pt, *private_pool_pt_end;
    };
    static ThreadBuffer tbuf;
    # pragma omp threadprivate(tbuf)

    void init_global(int _pool_size){
        pool_size = _pool_size;
        pool = new T[pool_size];
    }

    void clear_global(){
        shared_pool_pt = pool;
    }
    void clear_thread(){
        tbuf.private_pool_pt = tbuf.private_pool_pt_end = pool;
    }

    T* allocate(){
        if(tbuf.private_pool_pt < tbuf.private_pool_pt_end) return tbuf.private_pool_pt++;
        int allocate_size = buf_per_thread + rand() % randomized_buf_per_thread;
        tbuf.private_pool_pt = shared_pool_pt.fetch_add(allocate_size, memory_order_relaxed);
        tbuf.private_pool_pt_end = tbuf.private_pool_pt + allocate_size;
        
        #ifdef DEBUG
        if(tbuf.private_pool_pt_end - pool > pool_size) __printf("memory exceeded!!!");
        #endif
        return tbuf.private_pool_pt++;
    }
};

template<class T, class K, class HashFunc>
class ConcurrentHashMap
{
public:
    struct Node{
        T value;
        K key;
        Node* next;
    };

    struct HeadPointer{
        int pos;
        int version;
    };
    atomic<HeadPointer>* head_atomic;
    int head_verison;
    MultiThreadMemPool<Node>* pool;
    int head_size;
    HashFunc func;
    ConcurrentHashMap(int _head_size, MultiThreadMemPool<Node> *_pool){
        head_size = _head_size;
        head_atomic = new atomic<HeadPointer>[head_size];
        pool = _pool;
        func = HashFunc();
        head_verison = 0;
        for(int i = 0; i < head_size; i++) head_atomic[i].store({0, 0}, memory_order_relaxed);
    }
    void clear() {
        head_verison++;
    }
    bool test_valid(const HeadPointer &cur) {
        return cur.version == head_verison;
    }
    Node* get_point(const HeadPointer &cur) {
        return pool->pool + cur.pos;
    }
    HeadPointer store_point(Node* cur) {
        return {(int)(cur - pool->pool), head_verison};
    }
    T* get(const K &key, memory_order sync) {
        unsigned int idx = func(key) % head_size;

        HeadPointer cur_hp = head_atomic[idx].load(sync);
        Node* cur = nullptr;
        if(test_valid(cur_hp)){
            cur = get_point(cur_hp);
        }
        while(cur){
            if(cur->key == key) break;
            cur = cur->next;
        }
        if(cur == nullptr){
            return nullptr;
        }else{
            return &cur->value;
        }
    }

    T& get_or_create(const K &key, bool& create, memory_order sync){

        Node* cur = nullptr, *oricur = nullptr, *allo = nullptr;
        create = true;
        // #pragma omp critical
        {
            unsigned int idx = func(key) % head_size;
            HeadPointer cur_hp = head_atomic[idx].load(memory_order_acquire);
            do{
                if(test_valid(cur_hp)){
                    oricur = cur = get_point(cur_hp);
                }
                while(cur){
                    if(cur->key == key){
                        create = false;
                        return cur->value;
                    }
                    cur = cur->next;
                }
                if(allo == nullptr){
                    allo = pool->allocate();
                    allo->key = key;
                }
                allo->next = oricur;
            }while(!head_atomic[idx].compare_exchange_weak(cur_hp, store_point(allo), memory_order_acq_rel));
        }
        return allo->value;
    }
};

typedef pair<SearchNode*, int> HashKey;
typedef pair<int, int> HashNotifyKey;

typedef ConcurrentHashMap<float, HashKey, pair_hash> NodeStepMap;
typedef ConcurrentHashMap<SearchNode*, HashKey, pair_hash> NodeChildrenMap;
typedef ConcurrentHashMap<atomic<Notify*>, HashNotifyKey, pair_hash> NodeNotifyMap;
typedef SearchNode* SearchNode_pt;

extern int max_pos;
extern vector<pair<float, SearchNode*>>** beams;
extern NodeNotifyMap** node_notify_map_atomic;
extern NodeStepMap** node_step_map;

int query_vocab_index(char* word);
void global_init(int batch_size, int beam_size, int top_cand_n, int maxpos, int thread_num, char* lm_path);
void init_beam(int batch_size, int go_id);

template<class T>
void expand_beam(int batch_size, int step, T output_length, T dagscores, T nextstep_idx, T logits_idx, T lm_vocab, float top_p);

inline float& dagstep_get_or_create::operator()(int nextstep, bool &create, int batch_id, SearchNode* nextnode)
{
    return node_step_map[batch_id]->get_or_create(make_pair(nextnode, nextstep), create, memory_order_relaxed);
}