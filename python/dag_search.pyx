# distutils: language = c++
# cython: infer_types=True
# cython: language_level=3

cimport _kenlm
cimport cython
cimport openmp
import os
import numpy as np
import time
import sys
from libcpp.vector cimport vector
from libcpp cimport bool
from algorithm cimport nth_element, upper_bound, sort
from algorithm cimport max as cmax
from libc.math cimport INFINITY
from cython.operator cimport dereference as deref, preincrement as inc, address as addr
from cython.parallel import prange
from libcpp.utility cimport pair
from atomic cimport atomic, memory_order
cimport SearchBeam
from SearchBeam cimport __printf as printf
from SearchBeam cimport beams, node_notify_map_atomic
from SearchBeam cimport SearchNode, Notify, ExpandBeamCache, SearchNode_pt
from SearchBeam cimport init_beam, node_compare_allscore, calculate_score, make_pair, array_new, array_delete, expand_beam

init_time = 0
update_time = 0
expand_time = 0
lm_vocab = None
max_token = None

cdef bytes as_str(data):
    if isinstance(data, bytes):
        return data
    elif isinstance(data, unicode):
        return data.encode('utf8')
    raise TypeError('Cannot convert %s to string' % type(data))

def beam_search_init(int batch_size, int beam_size, int top_cand_n, int maxpos, int maxtoken, int threads_per_worker, tgt_dict, path=None):
    # Allocate memory and load vocabulary
    global lm_vocab
    global max_token
    lm_vocab = np.zeros(len(tgt_dict.symbols), dtype=np.intc)
    max_token = min(maxtoken, batch_size * maxpos)
    if path is not None:
        path = os.path.abspath(as_str(path))
        SearchBeam.global_init(batch_size, beam_size, top_cand_n, maxpos, max_token, threads_per_worker, path)
        #printf("load vocab start")
        for i, word in enumerate(tgt_dict.symbols):
            lm_vocab[i] = SearchBeam.query_vocab_index(as_str(word))
        #printf("load vocab end")
    else:
        SearchBeam.global_init(batch_size, beam_size, top_cand_n, maxpos, max_token, threads_per_worker, <char*>0)

@cython.boundscheck(False)
@cython.wraparound(False)
def dag_search(float[:, :, ::1] dagscores, int[:, :, ::1] nextstep_idx,
        int[:, :, ::1] logits_idx, int[::1] output_length,
        float alpha, float gamma, int beam_size, int beamlensize, float top_p, int pad_id, int go_id, int dedup,
        int no_consecutive_repeat_ngram, int no_repeat_ngram, int final_beam_size):

    batch_size = dagscores.shape[0]
    prelen = dagscores.shape[1]

    global init_time, update_time, expand_time, lm_vocab

    if SearchBeam.__debug_flag:
        printf("before init node\n")
        start_init = time.time()
    assert np.sum(output_length) < max_token
    init_beam(batch_size, go_id)
    if SearchBeam.__debug_flag:
        printf("after init node\n")
        init_time += time.time() - start_init

    cdef int i
    cdef int [::1] lm_vocab_view = lm_vocab

    for i in range(prelen):
        if SearchBeam.__debug_flag:
            printf("dag_search: i = %d\n", i)
            start = time.time()
        get_beam(batch_size, i, output_length, alpha, gamma, beam_size, beamlensize, final_beam_size)
        if SearchBeam.__debug_flag:
            printf("dag_search: finish get beam\n")
            start2 = time.time()
        expand_beam(batch_size, i, output_length, dagscores, nextstep_idx, logits_idx, lm_vocab_view, top_p, no_consecutive_repeat_ngram, no_repeat_ngram)
        if SearchBeam.__debug_flag:
            printf("dag_search: finish expand beam\n")
            start3 = time.time()
            update_time += start2 - start
            expand_time += start3 - start

    result = np.zeros((batch_size, final_beam_size, prelen), dtype=np.intc)
    score = np.zeros((batch_size, final_beam_size), dtype=np.float32)
    if SearchBeam.__debug_flag:
        printf("dag_search: before traverse\n")
    traverse_beam(batch_size, pad_id, result, score, dedup, final_beam_size)
    if SearchBeam.__debug_flag:
        printf("dag_search: after traverse\n")
        print(f"init_time {init_time} update_time {update_time}, expand_time {expand_time}")
    output_len = (result != pad_id).sum(axis=-1).max()
    return result[:, :, :output_len], score

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void get_beam(int batch_size, int step, int[::1] output_length,
    float alpha, float gamma, int beam_size, int beamlensize, int final_beam_size) nogil:

    cdef Notify* root
    cdef atomic[Notify*]* root_atomic
    cdef int i, j, now_beam_size, length, pid, tid, block
    cdef vector[pair[float, SearchNode_pt]]* beam
    cdef SearchNode* node
    cdef vector[SearchNode*]* beam_tmp
    cdef vector[int]* beam_cnt


    # step1: find all first beamlensize at (batch_id=i, length=j)

    block = step // 5 + 1

    for pid in prange(batch_size * block * 5, nogil=True, schedule="guided"):

        i = pid // (block * 5) # batch
        j = pid % (block * 5)  # step
        j = j // block + (j % block) * 5
        if j > step:
            continue

        beam = beams[i * SearchBeam.max_pos + j]
        if step < output_length[i]:
            beam.clear()

            now_beam_size = min(final_beam_size, beamlensize) if step == output_length[i] - 1 else beamlensize

            root_atomic = node_notify_map_atomic[i].get(make_pair(<int>step, <int>j), memory_order.memory_order_relaxed)
            if root_atomic == <atomic[Notify*]*>0:
                continue
            root = root_atomic.load(memory_order.memory_order_relaxed)
            # printf("getbeam batch=%d notify_root=%p\n", i, root)
            while root != <Notify*>0:
                beam.push_back(make_pair(calculate_score(root.target, alpha, gamma), <SearchNode_pt>root.target))
                root = root.next
            if (<int>beam.size()) > now_beam_size:
                nth_element(beam.begin(), beam.begin() + now_beam_size, beam.end(),
                     node_compare_allscore)
                beam.resize(now_beam_size)

        # printf("get_beam-step1-prange-end tid:%d pid:%d\n", tid, pid)

    # step2: find beamsize at batch=i
    for i in prange(batch_size, nogil=True, schedule="guided"):
        # tid = openmp.omp_get_thread_num()
        # printf("get_beam-step2-prange-start tid:%d pid:%d\n", tid, i)

        beam = beams[i * SearchBeam.max_pos]
        if step < output_length[i]:
            now_beam_size = final_beam_size if step == output_length[i] - 1 else beam_size

            for j in range(1, step + 1):
                beam.insert(beam.end(), beams[i * SearchBeam.max_pos + j].begin(), beams[i * SearchBeam.max_pos + j].end())
            if (<int>beam.size()) > now_beam_size:
                nth_element(beam.begin(), beam.begin() + now_beam_size, beam.end(),
                        node_compare_allscore)
                beam.resize(now_beam_size)
            if step == output_length[i] - 1:
                sort(beam.begin(), beam.end(), node_compare_allscore)
                

        if SearchBeam.__debug_flag:
            printf("getbeam finished, batch=%d beams:\n", i, )
            for j in range(<int>beam.size()):
                printf("\t[")
                node = deref(beam)[j].second
                SearchBeam.__debug_print_node(node)
                printf("] allscore=%f dagscore=%f lmscore=%f length=%d\n", deref(beam)[j].first, node.dagscore, node.lmscore, node.length)

        # printf("get_beam-step2-prange-end tid:%d pid:%d\n", tid, i)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void traverse_beam(int batch_size, int pad_id, int[:, :, ::1] result, float[:, ::1] score, int dedup, int final_beam_size) nogil:
    cdef int i, j
    cdef pair[float, SearchNode_pt] node_pair
    for i in prange(batch_size, nogil=True, schedule="guided"):
        for j in range(final_beam_size):
            if j < <int>deref(beams[i * SearchBeam.max_pos]).size():
                node_pair = deref(beams[i * SearchBeam.max_pos])[j]
                score[i, j] = node_pair.first
                traverse_beam_single(node_pair.second, i, pad_id, result, dedup, j)
            else:
                score[i, j] = -9999999999



@cython.wraparound(False)
@cython.boundscheck(False)
cdef void traverse_beam_single(SearchNode* beam, int batch, int pad_id, int[:, :, ::1] result, int dedup, int j) nogil:
    cdef int length = result.shape[2]
    cdef int pos, i

    pos = length - 1
    while beam != <SearchBeam.SearchNode*>0:
        result[batch, j, pos] = beam.word
        pos -= 1
        beam = beam.parent
    i = 0
    pos += 1
    while pos < length:
        if dedup > 0 and i > 0 and result[batch, j, i - 1] == result[batch, j, pos]:
            pos += 1
        else:
            result[batch, j, i] = result[batch, j, pos]
            i += 1
            pos += 1
    while i < length:
        result[batch, j, i] = pad_id
        i += 1
