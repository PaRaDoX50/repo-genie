import os
import sys
import pickle

import numpy as np
from faiss import IndexFlatL2
from db.sqlite_db_helper import SqliteDBHelper

from config.config import get_file_path, INDEX_CACHE_FILENAME, \
    INDEX_CACHE_PATH

TEXT_CHARS = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})


def is_text_file(filepath):
    return not bool(open(filepath, "rb").read(1024).translate(None, TEXT_CHARS))

def get_text_files(directory=".", ignore_paths=[]):
    text_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in ignore_paths]
        for i, filename in enumerate(files, start=1):
            filepath = os.path.join(root, filename)
            if (
                os.path.isfile(filepath)
                and not any(ignore_path in filepath for ignore_path in ignore_paths)
                and is_text_file(filepath)
            ):
                text_files.append(filepath)
    return text_files


def get_files_with_contents(directory, ignore_paths, cache_db, verbose):
    text_files = get_text_files(directory, ignore_paths)
    files_with_contents = []
    cache = SqliteDBHelper(cache_db)
    for filepath in text_files:
        file_stat = os.stat(filepath)

        query = f"SELECT * FROM unnamed WHERE key = ?"
        result = cache.getOne(query, (filepath,))

        file_info_cache = None
        if result is not None:
            file_info_cache = pickle.loads(result[1])
            
        if file_info_cache and file_info_cache["mtime"] == file_stat.st_mtime:
            file_info_cache["fetchFromCache"] = True
            files_with_contents.append(file_info_cache)
        else:
            file_info_cache = {
                "filepath": os.path.abspath(filepath),
                "mtime": file_stat.st_mtime,
            }
            cache[filepath] = file_info_cache
            file_info_cache["fetchFromCache"] = False
            files_with_contents.append(file_info_cache)

    return files_with_contents


def create_file_index(
    embed, ignore_paths, embed_chunk_size, extra_dirs=[], verbose=False
):
    cache_db = get_file_path(INDEX_CACHE_PATH, INDEX_CACHE_FILENAME)

    files_with_contents = []

    # Add files from additional folders
    for folder in extra_dirs:
        if os.path.exists(folder):
            folder_files = get_files_with_contents(
                folder, ignore_paths, cache_db, verbose
            )
            files_with_contents.extend(folder_files)
        else:
            print(f"Warning: Additional folder {folder} does not exist\n")

    if not files_with_contents:
        return None, []

    chunks = []
    embeddings_list = []
    files = 0
    for file_info in files_with_contents:
        files += 1
        print(f"Indexing files progress: {files}/{len(files_with_contents)}")
        filepath = file_info["filepath"]
        if file_info["fetchFromCache"]:
            filepath = file_info["filepath"]
            cached_chunks = get_all_file_chunks_and_embeddings(cache_db, filepath)
            if cached_chunks and len(cached_chunks) > 0:
                if verbose:
                    print("Getting chunks and embeddings from cache for file: ", filepath)
                chunks_from_cache = [i["chunk"] for i in cached_chunks]
                embeddings_from_cache = [i["embedding"] for i in cached_chunks]

                chunks.extend(chunks_from_cache)
                embeddings_list.extend(embeddings_from_cache)
                continue
        
        try:
            with open(filepath, "r") as file:
                contents = file.read()
        except UnicodeDecodeError:
            print(f"Skipping {filepath} because it is not a text file.\n")
            continue

        file_chunks, file_embeddings = process_file(embed, filepath, contents, embed_chunk_size, cache_db)
        chunks.extend(file_chunks)
        embeddings_list.extend(file_embeddings)
            
    embeddings = np.array(embeddings_list)
    index = IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks


def process_file(embed, filepath, contents, embed_chunk_size, cache_db, verbose=False):
    lines = contents.split("\n")
    current_chunk = ""
    start_line_number = 1
    chunks = []
    embeddings_list = []
    cache = SqliteDBHelper(cache_db)
    if verbose:
        print(f"Creating embeddings for {filepath}\n")
    chunk_num = 0
    for line_number, line in enumerate(lines, start=1):
        # Process each line individually if needed
        line_content = line
        while line_content:
            proposed_chunk = current_chunk + line_content + "\n"
            chunk_header = f"---------------\n\nUser file '{filepath}' lines {start_line_number}-{line_number}:\n\n"
            proposed_text = chunk_header + proposed_chunk
            chunk_tokens = embed.count_tokens(proposed_text)
            if chunk_tokens <= embed_chunk_size:
                current_chunk = proposed_chunk
                break  # The line fits in the current chunk, break out of the inner loop
            else:
                # Split line if too large for a new chunk
                if current_chunk == "":
                    split_point = find_split_point(
                        embed, line_content, embed_chunk_size, chunk_header
                    )
                    current_chunk = line_content[:split_point] + "\n"
                    line_content = line_content[split_point:]
                else:
                    # Save the current chunk as it is, and start a new one
                    final_chunk = {
                        "tokens": embed.count_tokens(chunk_header + current_chunk),
                        "text": chunk_header + current_chunk,
                        "filepath": filepath,
                        "chunk_num": chunk_num
                    }

                    chunks.append(final_chunk)

                    embedding = embed.create_embedding(chunk_header + current_chunk)
                    embeddings_list.append(embedding)
                    query = "insert into unnamed (key, value) values (?, ?)"
                    cache.save(query, (f"{filepath}:chunk:{chunk_num}", pickle.dumps({
                        "chunk": final_chunk,
                        "embedding": embedding
                    })))
                    chunk_num += 1
                    current_chunk = ""
                    start_line_number = line_number  # Next chunk starts from this line
                    # Do not break; continue processing the line
                    # Add this line to prevent infinite loops
                    line_content = line_content.strip()  # Ensure there is actually some remaining string
                    if not line_content:
                        break
    # Add the remaining content as the last chunk
    if current_chunk:
        chunk_header = f"---------------\n\nUser file '{filepath}' lines {start_line_number}-{len(lines)}:\n\n"

        final_chunk = {
            "tokens": embed.count_tokens(chunk_header + current_chunk),
            "text": chunk_header + current_chunk,
            "filepath": filepath,
            "chunk_num": chunk_num
        }

        chunks.append(final_chunk)

        embedding = embed.create_embedding(chunk_header + current_chunk)
        embeddings_list.append(embedding)

        cache.save("insert into unnamed (key, value) values (?, ?)", (f"{filepath}:chunk:{chunk_num}", pickle.dumps({
            "chunk": final_chunk,
            "embedding": embedding
        })))
        chunk_num += 1
    return chunks, embeddings_list


def find_split_point(embed, line_content, max_size, header):
    low = 0
    high = len(line_content)
    while low < high:
        mid = (low + high) // 2
        if embed.count_tokens(header + line_content[:mid] + "\n") < max_size:
            low = mid + 1
        else:
            high = mid
    return low - 1


def search_index(embed, index, query, all_chunks):
    query_embedding = embed.create_embedding(query)
    try:
        distances, indices = index.search(
            np.array([query_embedding]), 100
        )
    except AssertionError as e:
        raise e
    relevant_chunks = [all_chunks[i] for i in indices[0] if i != -1]
    return relevant_chunks

def get_all_file_chunks_and_embeddings(cache_db, file_name):
    cache = SqliteDBHelper(cache_db)
    result = cache.get("SELECT * FROM unnamed WHERE key LIKE ?", (f"{file_name}:chunk:%",))
    return [pickle.loads(res[1]) for res in result]