"""
T3P file reading module for tpx3HitParser.

This module provides functions to read T3P binary files and convert them
to the same data structures used by the TPX3 parser.

T3P files have a simple structure: each packet is 16 bytes containing
Index (4 bytes), ToA (8 bytes), Overflow (1 byte), ftoa (1 byte), ToT (2 bytes).
"""

import logging
import struct
import numpy as np
import os

from lib.constants import dt_hit

logger = logging.getLogger('root')

# T3P format constants
T3P_PACKET_SIZE = 16  # bytes per packet
T3P_FINETICK_NS = 1.5625  # nanoseconds per ToA tick
T3P_TICK_NS = 25  # nanoseconds per ToT tick


def read_t3p_positions(f):
    """
    Generator that yields file positions for T3P data chunks.
    
    T3P files have a simple structure: each packet is 16 bytes containing
    Index (4 bytes), ToA (8 bytes), Overflow (1 byte), ftoa (1 byte), ToT (2 bytes).
    
    Args:
        f: Open file handle for T3P file
        
    Yields:
        List of [file_position, chunk_size_bytes, chip_id, rollover_counter]
        where chip_id is always 0 for T3P files and rollover_counter is 0
    """
    file_size = os.fstat(f.fileno()).st_size
    
    # Check that file size is compatible with packet structure
    if file_size % T3P_PACKET_SIZE != 0:
        logger.warning(f"T3P file size {file_size} is not a multiple of packet size {T3P_PACKET_SIZE}")
    
    # Calculate number of packets
    n_packets = file_size // T3P_PACKET_SIZE
    
    # For T3P files, we can process the entire file as one chunk
    # or split into smaller chunks for memory efficiency
    chunk_size_packets = 1000000  # Process 1M packets at a time
    chunk_size_bytes = chunk_size_packets * T3P_PACKET_SIZE
    
    position = 0
    while position < file_size:
        # Calculate actual chunk size (last chunk might be smaller)
        actual_chunk_size = min(chunk_size_bytes, file_size - position)
        
        yield [position, actual_chunk_size, 0, 0]  # chip_id=0, rollover_counter=0
        position += actual_chunk_size


def parse_t3p_data_packages(positions, f, tot_correction, settings):
    """
    Parse T3P data packages into hits array.
    
    Args:
        positions: List of file positions from read_t3p_positions()
        f: Open file handle for T3P file
        tot_correction: ToT correction matrix (not used for T3P)
        settings: Settings object with parsing parameters
        
    Returns:
        Tuple of (hits_array, min_toa, max_toa)
    """
    # Calculate total number of hits
    n_hits = sum(pos[1] // T3P_PACKET_SIZE for pos in positions)
    hits = np.zeros(n_hits, dtype=dt_hit)
    
    logger.info(f"Parsing {n_hits} T3P hits")
    
    i = 0
    min_toa = np.iinfo(np.int64).max
    max_toa = np.iinfo(np.int64).min
    
    for pos in positions:
        chunk_hits = parse_t3p_data_package(f, pos, tot_correction, settings.hits_tot_threshold)
        
        for hit in chunk_hits:
            if hit is not None:
                hits[i] = hit
                min_toa = min(min_toa, hit[4])  # ToA field
                max_toa = max(max_toa, hit[4])  # ToA field
                i += 1
    
    # Resize array to actual number of hits processed
    hits.resize((i,), refcheck=False)
    
    logger.info(f"Successfully parsed {i} T3P hits")
    logger.info(f"ToA range: {min_toa} to {max_toa}")
    
    return hits, min_toa, max_toa


def parse_t3p_data_package(f, pos, tot_correction, tot_threshold):
    """
    Parse a single T3P data package chunk.
    
    Args:
        f: Open file handle for T3P file
        pos: Position tuple [file_position, chunk_size_bytes, chip_id, rollover_counter]
        tot_correction: ToT correction matrix (not used for T3P)
        tot_threshold: Minimum ToT threshold
        
    Yields:
        Hit tuples: (chipId, x, y, ToT, ToA)
    """
    file_pos, chunk_size, chip_id, rollover_counter = pos
    
    # Seek to the correct position
    f.seek(file_pos)
    
    # Read the chunk
    data = f.read(chunk_size)
    if len(data) != chunk_size:
        logger.warning(f"Could not read full chunk at position {file_pos}")
        return
    
    # Calculate number of packets in this chunk
    n_packets = len(data) // T3P_PACKET_SIZE
    
    for i in range(n_packets):
        packet_start = i * T3P_PACKET_SIZE
        packet_data = data[packet_start:packet_start + T3P_PACKET_SIZE]
        
        if len(packet_data) < T3P_PACKET_SIZE:
            logger.warning(f"Incomplete packet at position {file_pos + packet_start}")
            break
        
        # Unpack the packet: Index (4), ToA (8), Overflow (1), ftoa (1), ToT (2)
        index, toa_raw, overflow, ftoa, tot_raw = struct.unpack('<IQbbH', packet_data)
        
        # Apply ToT threshold filter
        if tot_raw < tot_threshold:
            continue
        
        # Extract pixel coordinates from index
        pixel_idx = index & 0xFFFF  # Lower 16 bits contain pixel index
        x = pixel_idx % 256
        y = pixel_idx // 256
        
        # Calculate ToA in nanoseconds
        toa_ns = int(toa_raw * T3P_TICK_NS - ftoa * T3P_FINETICK_NS)
        
        # Yield hit tuple: (chipId, x, y, ToT, ToA)
        yield (chip_id, x, y, tot_raw, toa_ns)


def check_t3p_file(file_path):
    """
    Check if a file is a valid T3P file.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file appears to be a valid T3P file, False otherwise
    """
    try:
        file_size = os.path.getsize(file_path)
        
        # Check file size is compatible with packet structure
        if file_size % T3P_PACKET_SIZE != 0:
            return False
        
        # Check that file is not empty
        if file_size == 0:
            return False
        
        return True
        
    except (OSError, IOError):
        return False


def calculate_t3p_image_shape(file_path):
    """
    Calculate the image shape for a T3P file.
    
    Args:
        file_path: Path to the T3P file
        
    Returns:
        Tuple of (width, height) for the image
    """
    # T3P files are always single chip, 256x256
    return (256, 256)


def read_t3p_file_info(file_path):
    """
    Read basic information about a T3P file.
    
    Args:
        file_path: Path to the T3P file
        
    Returns:
        Dictionary with file information
    """
    try:
        file_size = os.path.getsize(file_path)
        n_packets = file_size // T3P_PACKET_SIZE
        
        return {
            "file_path": file_path,
            "file_size_bytes": file_size,
            "packet_size_bytes": T3P_PACKET_SIZE,
            "n_packets": n_packets,
            "format": "T3P",
            "chip_count": 1,
            "image_shape": (256, 256)
        }
        
    except (OSError, IOError) as e:
        logger.error(f"Error reading T3P file info: {e}")
        return None
