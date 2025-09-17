"""
ADVS file reading module for tpx3HitParser.

This module provides functions to read ADVS binary files and convert them
to the same data structures used by the TPX3 parser.
"""

import logging
import struct
import numpy as np
import os

from lib.constants import dt_hit

logger = logging.getLogger('root')

# ADVS format constants
ADVS_PACKET_SIZE = 16  # bytes per packet
ADVS_FINETICK_NS = 1.5625  # nanoseconds per ToA tick
ADVS_TICK_NS = 25  # nanoseconds per ToT tick


def read_advs_positions(f):
    """
    Generator that yields file positions for ADVS data chunks.
    
    ADVS files have a simple structure: each packet is 16 bytes containing
    Address (4 bytes), ToT (4 bytes), and ToA (8 bytes).
    
    Args:
        f: Open file handle for ADVS file
        
    Yields:
        List of [file_position, chunk_size_bytes, chip_id, rollover_counter]
        where chip_id is always 0 for ADVS files and rollover_counter is 0
    """
    file_size = os.fstat(f.fileno()).st_size
    
    # Check that file size is compatible with packet structure
    if file_size % ADVS_PACKET_SIZE != 0:
        logger.warning(f"ADVS file size {file_size} is not a multiple of packet size {ADVS_PACKET_SIZE}")
    
    # Calculate number of packets
    n_packets = file_size // ADVS_PACKET_SIZE
    
    # For ADVS files, we can process the entire file as one chunk
    # or split into smaller chunks for memory efficiency
    chunk_size_packets = 1000000  # Process 1M packets at a time
    chunk_size_bytes = chunk_size_packets * ADVS_PACKET_SIZE
    
    position = 0
    while position < file_size:
        remaining_bytes = file_size - position
        current_chunk_size = min(chunk_size_bytes, remaining_bytes)
        
        # Ensure chunk size is aligned to packet boundaries
        current_chunk_size = (current_chunk_size // ADVS_PACKET_SIZE) * ADVS_PACKET_SIZE
        
        if current_chunk_size > 0:
            yield [position, current_chunk_size, 0, 0]  # chip_id=0, rollover=0 for ADVS
            position += current_chunk_size
        else:
            break


def parse_advs_data_packages(positions, f, tot_correction, settings):
    """
    Parse ADVS data packages into hits array.
    
    Args:
        positions: List of file positions from read_advs_positions()
        f: Open file handle for ADVS file
        tot_correction: ToT correction matrix (not used for ADVS)
        settings: Settings object with parsing parameters
        
    Returns:
        Tuple of (hits_array, min_toa, max_toa)
    """
    # Calculate total number of hits
    n_hits = sum(pos[1] // ADVS_PACKET_SIZE for pos in positions)
    hits = np.zeros(n_hits, dtype=dt_hit)
    
    logger.info(f"Parsing {n_hits} ADVS hits")
    
    i = 0
    min_toa = np.iinfo(np.int64).max
    max_toa = np.iinfo(np.int64).min
    
    for pos in positions:
        chunk_hits = parse_advs_data_package(f, pos, tot_correction, settings.hits_tot_threshold)
        
        for hit in chunk_hits:
            if hit is not None:
                hits[i] = hit
                min_toa = min(min_toa, hit[4])  # ToA field
                max_toa = max(max_toa, hit[4])  # ToA field
                i += 1
    
    # Resize array to actual number of hits processed
    hits.resize((i,), refcheck=False)
    
    logger.info(f"Successfully parsed {i} ADVS hits")
    logger.info(f"ToA range: {min_toa} to {max_toa}")
    
    return hits, min_toa, max_toa


def parse_advs_data_package(f, pos, tot_correction, tot_threshold):
    """
    Parse a single ADVS data package chunk.
    
    Args:
        f: Open file handle for ADVS file
        pos: Position tuple [file_position, chunk_size_bytes, chip_id, rollover_counter]
        tot_correction: ToT correction matrix (not used for ADVS)
        tot_threshold: Minimum ToT threshold
        
    Yields:
        Hit tuples (chipId, x, y, ToT, ToA) or None for filtered hits
    """
    f.seek(pos[0])
    data = f.read(pos[1])
    
    # Calculate number of packets in this chunk
    n_packets = pos[1] // ADVS_PACKET_SIZE
    
    # Process each packet individually
    for i in range(n_packets):
        packet_start = i * ADVS_PACKET_SIZE
        packet_data = data[packet_start:packet_start + ADVS_PACKET_SIZE]
        
        try:
            address, tot_raw, toa_raw = struct.unpack('<IIQ', packet_data)
        except struct.error as e:
            logger.error(f'Failed reading ADVS packet {i} at position {pos[0] + packet_start} (error: {e})')
            continue
        
        # Extract coordinates from address (lower 16 bits contain pixel index)
        pixel_idx = address & 0xFFFF
        x = pixel_idx % 256
        y = pixel_idx // 256
        
        # Extract ToT (only lower 16 bits are valid)
        tot = tot_raw & 0xFFFF
        
        # Apply ToT threshold filter
        if tot < tot_threshold:
            yield None
            continue
        
        # For ADVS files, we use raw ToA values directly
        # The time conversion can be done later if needed
        toa = toa_raw
        
        # Yield hit tuple: (chipId, x, y, ToT, ToA)
        yield (pos[2], x, y, tot, toa)  # pos[2] is chip_id (always 0 for ADVS)


def check_advs_file(file_path):
    """
    Check if file is a valid ADVS file.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if valid, error message string if invalid
    """
    if not os.path.exists(file_path):
        return f"ADVS file {file_path} does not exist"
    
    file_size = os.path.getsize(file_path)
    
    if file_size % ADVS_PACKET_SIZE != 0:
        return f"ADVS file size {file_size} is not compatible with packet size {ADVS_PACKET_SIZE}"
    
    # Try to read first packet to verify format
    try:
        with open(file_path, 'rb') as f:
            data = f.read(ADVS_PACKET_SIZE)
            if len(data) < ADVS_PACKET_SIZE:
                return f"ADVS file {file_path} is too small to contain even one packet"
            
            # Unpack first packet
            address, tot, toa = struct.unpack('<IIQ', data)
            
            # Basic sanity checks
            pixel_idx = address & 0xFFFF
            x = pixel_idx % 256
            y = pixel_idx // 256
            
            if x >= 256 or y >= 256:
                return f"ADVS file {file_path} contains invalid pixel coordinates"
            
            if tot > 0xFFFF:
                return f"ADVS file {file_path} contains invalid ToT value"
            
            logger.info(f"ADVS file {file_path} validation passed")
            logger.info(f"  File size: {file_size} bytes ({file_size // ADVS_PACKET_SIZE} packets)")
            logger.info(f"  First packet: addr={address} x={x} y={y} ToT={tot} ToA={toa}")
            
    except Exception as e:
        return f"ADVS file {file_path} validation failed: {e}"
    
    return True


def calculate_advs_image_shape():
    """
    Calculate image shape for ADVS files.
    
    Returns:
        Image shape tuple (width, height)
    """
    return (256, 256)  # ADVS files are always 256x256 pixels
