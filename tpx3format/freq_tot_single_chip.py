import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
import lib

logger = logging.getLogger('root')


def build_freq_tot(hits, cores):
    logger.info("Determine and store ToT frequency matrix (single chip)")
    
    # Determine number of chips from data
    max_chip_id = np.max(hits['chipId'])
    num_chips = max_chip_id + 1
    total_pixels = num_chips * 256 * 256
    
    logger.info(f"Detected {num_chips} chip(s), total pixels: {total_pixels}")

    # Setup pool
    pool = multiprocessing.Pool(cores)
    results = {}

    # Progress bar
    progress_bar = tqdm(total=len(hits), unit="hits", smoothing=0.1, unit_scale=True)

    # First split hits in chunks
    chunk_size = int(len(hits) / cores / 5)
    if chunk_size > len(hits):
        logger.warning("Hits chunk size is larger than amount of hits")
        chunk_size = len(hits)

    start = 0
    r = 0
    while start < len(hits):
        end = start + chunk_size

        if end > len(hits):
            end = len(hits)

        hits_chunk = hits[start:end]
        results[r] = pool.apply_async(calc_freq_tot, args=[hits_chunk, total_pixels])
        start = end
        r += 1

    pool.close()

    freq_tot = np.zeros((total_pixels, 1024), dtype='uint32')

    for idx in range(0, len(results)):
        freq_tot_chunk = results[idx].get()
        progress_bar.update(chunk_size)
        freq_tot += freq_tot_chunk
        del results[idx]

    progress_bar.close()
    return freq_tot


def calc_freq_tot(hits, total_pixels):
    vector_idx = hits['chipId'] * (256 * 256) + hits['x'] + hits['y'] * 256
    
    freq_tot = np.zeros((total_pixels, 1024), dtype='uint16')

    for idx, hit in enumerate(hits):
        if vector_idx[idx] < total_pixels:  # Safety check
            freq_tot[vector_idx[idx]][hit['ToT']] += 1

    return freq_tot
