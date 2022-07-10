from atomic cimport atomic
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from atomic cimport memory_order
from libcpp cimport bool
cimport _kenlm 

cdef extern from *:
    """
    template <typename T>
    T* array_new(int n) {
        return new T[n];
    }

    template <typename T>
    void array_delete(T* x) {
        delete [] x;
    }
    """
    T* array_new[T](int)
    void array_delete[T](T* x)

cdef extern from "<utility>" namespace "std" nogil:
   pair[ T, U ] make_pair[ T, U ]( T&, U& ) nogil

cdef extern from "python/SearchBeam.h":

    cdef struct SearchNode
    ctypedef SearchNode* SearchNode_pt

    cdef cppclass QuickMap[K, V]:
        V& get_or_create(const K &k, bool &create, int batch, SearchNode_pt node) nogil

    cdef struct SearchNode:
        SearchNode *parent
        int word, length
        _kenlm.State lm_state
        float lmscore, dagscore
        QuickMap[int, float] dagstepscore_map

    cdef struct Notify:
        SearchNode *target
        Notify* next

    cdef cppclass ExpandBeamCache:
        SearchNode* cached_nextnode
        float cached_add_score
        SearchNode* search_node
        int search_nextword

        ExpandBeamCache() nogil
        SearchNode* load(int batch, SearchNode* node, int nextword, int lm_word) nogil
        void write_back() nogil
        void addscore(float dagscore) nogil

    cdef cppclass ConcurrentHashMap[T, K, HashFunc]:
        T* get(const K &key, memory_order sync) nogil
        T& get_or_create(K &key, bool &create, memory_order sync) nogil
    cdef cppclass HashFunc:
        pass
    ctypedef pair[SearchNode_pt, int] HashKey
    ctypedef pair[int, int] HashNotifyKey
    ctypedef ConcurrentHashMap[float, HashKey, HashFunc] NodeStepMap
    ctypedef ConcurrentHashMap[atomic[Notify*], HashNotifyKey, HashFunc] NodeNotifyMap

    cdef bool __debug_flag
    cdef int max_pos
    cdef vector[pair[float, SearchNode_pt]]** beams
    cdef NodeNotifyMap** node_notify_map_atomic
    cdef NodeStepMap** node_step_map

    cdef int query_vocab_index(char* word) nogil
    cdef bool node_compare_allscore(const pair[float, SearchNode*] &a, const pair[float, SearchNode*] &b) nogil
    cdef float calculate_score(SearchNode* node, float alpha, float gamma) nogil

    cdef void global_init(int batch_size, int beam_size, int top_cand_n, int maxpos, int thread_num, char* lm_path) nogil
    cdef void init_beam(int batch_size, int go_id) nogil
    cdef void add_step_dagscore(int batch, SearchNode* nextnode_readonly, int nextstep, float dagscore) nogil
    cdef void expand_beam(int batch_size, int step, int[::1] output_length, float[:, :, ::1] dagscores, int[:, :, ::1] nextstep_idx, int[:, :, ::1] logits_idx, int [::1] lm_vocab, float top_p) nogil

    cdef void __debug_print_node(SearchNode* now) nogil
    cdef int __printf(const char *template, ...) nogil