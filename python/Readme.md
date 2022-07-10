We recommand to read the appendix of DA-Transformer paper before reading the codes.

Here is a breif explanation of the algorithm implementation:

```
1. Init the search beam (init_beam)
    1.1  Create a start node.
    1.2  Insert a notify at step 0.  (node_notify_map_atomic)
2. For step i
    2.1  Find all beams (get_beam)
        2.1.1 enumerate all notifies in node_notify_map_atomic to get all active beams
        2.1.2 we sort the beams according to their scores. We limit the number of beams (two stage filter, beamlensize and beamsize)
    2.2  Expand beams (expand_beam)
        2.2.1 we first get the now_node, indicating the current beam.
        2.2.2 we get the beam score from now_node->dagstepscore_map, using step as the query key. 
                Note one beam (indicating the paths that have the same prefix) may appear at different steps
        2.2.3 we enumerate the next transition and invoke expand_path
            2.2.3.1  we use thread_expand_cache.load to get or create the next node
            2.2.3.2  we add the score to the node (but we do not write to memory right away, which may cause conflicts for multi threads.
                            We want to merge all the write operations.)
            2.2.3.3  we insert a notify in the list, which records the score. It will be used in find the max beams.
3. Find the max beam (traverse_beam)
```
