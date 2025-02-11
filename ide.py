#!/usr/bin/env python3
import os
import subprocess
import json
import requests
import time
import re
import difflib
import threading
import ast
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np  # For cosine similarity and vector math
import patch  # python-patch library for applying unified diffs

# Third-party library for auto file indexing; install via pip: watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("[Warning] watchdog module not found. Automatic index refresh will be disabled.")
    Observer = None

# --- Configuration Management ---
# Default configuration values.
DEFAULT_CONFIG = {
    "OLLAMA_BASE_URL": "http://localhost:11434",  # Base URL for the Ollama API server
    "CHAT_MODEL": "llama3.1:8b-instruct-q2_K",                        # Chat model used for conversation and diff generation
    "EMBED_MODEL": "paraphrase-multilingual",            # Embedding model name used to generate embeddings
    "DEFAULT_PROJECT_ROOT": r"C:\Users\Llew\Desktop\ide-ollama\ide dev env",     # Default project directory
    "FRAMEWORK_FILE": "framework.json",            # JSON file that describes the codebase framework
    "INDEX_FILE_EXTENSIONS": [".py", ".txt", ".json"],  # File extensions to index
    "EMBEDDING_CACHE_FILE": "embedding_cache.json",   # Persistent embedding cache file
    "CHANGE_LOG_FILE": "change_log.txt"
}

# Load configuration from file if available.
def load_config(config_path="config.json"):
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"[Config] Loaded configuration from {config_path}.")
            return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"[Config] Error reading config file: {e}. Using default configuration.")
    return DEFAULT_CONFIG

CONFIG = load_config()

OLLAMA_BASE_URL = CONFIG["OLLAMA_BASE_URL"]
CHAT_MODEL = CONFIG["CHAT_MODEL"]
EMBED_MODEL = CONFIG["EMBED_MODEL"]
DEFAULT_PROJECT_ROOT = Path(CONFIG["DEFAULT_PROJECT_ROOT"])
FRAMEWORK_FILE = Path(CONFIG["FRAMEWORK_FILE"])
INDEX_FILE_EXTENSIONS = CONFIG["INDEX_FILE_EXTENSIONS"]
EMBEDDING_CACHE_FILE = Path(CONFIG["EMBEDDING_CACHE_FILE"])
CHANGE_LOG_FILE = CONFIG["CHANGE_LOG_FILE"]

# --- Persistent Embedding Cache ---
# In-memory cache that will be synced with disk.
_embedding_cache = {}

def load_embedding_cache():
    global _embedding_cache
    if EMBEDDING_CACHE_FILE.exists():
        try:
            with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
                _embedding_cache = json.load(f)
            print(f"[Cache] Loaded embedding cache from {EMBEDDING_CACHE_FILE}.")
        except Exception as e:
            print(f"[Cache] Failed to load embedding cache: {e}")
            _embedding_cache = {}
    else:
        _embedding_cache = {}

def save_embedding_cache():
    try:
        with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_embedding_cache, f)
        print(f"[Cache] Saved embedding cache to {EMBEDDING_CACHE_FILE}.")
    except Exception as e:
        print(f"[Cache] Failed to save embedding cache: {e}")

def clear_embedding_cache():
    """Clear the persistent embedding cache both in memory and on disk."""
    global _embedding_cache
    _embedding_cache = {}
    if EMBEDDING_CACHE_FILE.exists():
        try:
            EMBEDDING_CACHE_FILE.unlink()
            print(f"[Cache] Embedding cache file {EMBEDDING_CACHE_FILE} removed.")
        except Exception as e:
            print(f"[Cache] Failed to remove cache file: {e}")
    print("[Cache] Embedding cache cleared.")

load_embedding_cache()

# --- API Helper Functions ---

def get_embedding(text, max_retries=3, retry_delay=2):
    """
    Retrieve the embedding for the given text using Ollama's embedding API with retry and caching.
    """
    if not text:
        raise ValueError("Input text for embedding must not be empty.")

    if text in _embedding_cache:
        return _embedding_cache[text]

    url = f"{OLLAMA_BASE_URL}/api/embed"
    data = {"model": EMBED_MODEL, "input": text, "truncate": True}
    attempt = 0

    while attempt < max_retries:
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            embedding = response.json()["embeddings"][0]
            _embedding_cache[text] = embedding
            save_embedding_cache()
            return embedding
        except requests.RequestException as e:
            attempt += 1
            if attempt >= max_retries:
                raise requests.HTTPError(f"Failed to get embedding after {max_retries} attempts: {e}")
            time.sleep(retry_delay)

def embed_file_content(file_path, framework_file_path=FRAMEWORK_FILE):
    """
    Embed every line of a file and update the framework file with embeddings.
    """
    framework = {"files": []}

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    file_embeddings = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        try:
            embedding = get_embedding(line)
            file_embeddings.append({"line": i + 1, "text": line, "embedding": embedding})
        except requests.HTTPError as e:
            print(f"[Embedding Error] Failed to embed line {i + 1} in {file_path}: {e}")
    
    file_info = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "line_count": len(lines),
        "embeddings": file_embeddings
    }

    framework["files"].append(file_info)

    try:
        with open(framework_file_path, "w", encoding="utf-8") as f:
            json.dump(framework, f, indent=4)
        print(f"[Framework] Updated framework file: {framework_file_path}")
    except Exception as e:
        print(f"[Framework] Failed to update framework file: {e}")

def generate_chat_response(messages, stream=True, max_retries=3, retry_delay=2, generate_patch=False):
    """
    Generate a chat response from the Ollama chat API based on a list of messages.
    Supports streaming if requested.
    
    If `generate_patch` is True, it will extract the patch from the generated unified diff.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {"model": CHAT_MODEL, "messages": messages, "stream": stream}
    attempt = 0

    while attempt < max_retries:
        try:
            if stream:
                response = requests.post(url, json=payload, stream=True)
                response.raise_for_status()
                full_content = ""
                # Stream and print tokens as they arrive.
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            token = json.loads(line)
                            token_content = token.get("message", {}).get("content", "")
                            print(token_content, end="", flush=True)
                            full_content += token_content
                        except json.JSONDecodeError:
                            continue
                print()  # Newline after streaming.

                # If we need to generate a patch from the diff
                if generate_patch and full_content.startswith("--- "):
                    return generate_patch_from_diff(full_content)

                return full_content
            else:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                full_response = response.json()["message"]["content"]

                # If we need to generate a patch from the diff
                if generate_patch and full_response.startswith("--- "):
                    return generate_patch_from_diff(full_response)

                return full_response
        except requests.RequestException as e:
            attempt += 1
            if attempt >= max_retries:
                raise requests.HTTPError(f"Chat API request failed after {max_retries} attempts: {e}")
            time.sleep(retry_delay)


def generate_patch_from_diff(diff_text):
    """
    Converts a unified diff into an actual patched code.
    This function extracts the diff and applies it to generate modified code.
    """
    lines = diff_text.splitlines()
    patched_code = []
    in_diff = False

    for line in lines:
        if line.startswith("--- ") or line.startswith("+++ "):
            in_diff = True  # Start of the diff block
            continue
        if in_diff:
            if line.startswith("-"):
                continue  # Remove deleted lines
            elif line.startswith("+"):
                patched_code.append(line[1:])  # Add modified lines
            else:
                patched_code.append(line)  # Keep unchanged lines

    return "\n".join(patched_code)

# --- Code Base Indexing and File Search ---

class CodeIndex:
    """
    Manage an in-memory index of a code base.
    The index maps file paths to their content and embedding vectors.
    """
    def __init__(self, project_root, file_extensions=INDEX_FILE_EXTENSIONS):
        self.project_root = Path(project_root)
        self.file_extensions = file_extensions
        self.index = {}  # Mapping: {Path: {"content": str, "embedding": list}}

    def index_file(self, file_path):
        """
        Index a single file by reading its content and computing its embedding.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"[Index] Error reading file {file_path}: {e}")
            return

        try:
            embedding = get_embedding(content)
        except Exception as e:
            print(f"[Index] Error getting embedding for {file_path}: {e}")
            embedding = []
        self.index[file_path] = {"content": content, "embedding": embedding}
        print(f"[Index] Indexed: {file_path}")

    def build_index(self):
        """
        Recursively traverse the project directory and index all files with allowed extensions.
        """
        print(f"[Index] Building index for project: {self.project_root}")
        for file_extension in self.file_extensions:
            for file_path in self.project_root.rglob(f"*{file_extension}"):
                self.index_file(file_path)
        print(f"[Index] Completed. Total files indexed: {len(self.index)}")
        self.update_framework_file()

    def refresh_index(self):
        """
        Refresh the index by rescanning the project directory, adding new files and
        removing files that no longer exist.
        """
        print("[Index] Refreshing code index...")
        current_files = set()
        for file_extension in self.file_extensions:
            current_files.update(set(self.project_root.rglob(f"*{file_extension}")))

        # Remove files that no longer exist.
        removed_files = set(self.index.keys()) - current_files
        for file_path in removed_files:
            print(f"[Index] Removing file from index: {file_path}")
            self.index.pop(file_path, None)

        # Index new or modified files.
        for file_path in current_files:
            if file_path not in self.index:
                self.index_file(file_path)
            else:
                # Optionally, update files that have changed on disk.
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content != self.index[file_path]["content"]:
                        print(f"[Index] Detected change in file: {file_path}")
                        self.index_file(file_path)
                except Exception as e:
                    print(f"[Index] Error updating file {file_path}: {e}")
        self.update_framework_file()
        print("[Index] Refresh complete.")

    def update_file(self, file_path):
        """
        Update the index for a file after it has been modified.
        """
        print(f"[Index] Updating file: {file_path}")
        self.index_file(file_path)
        self.update_framework_file()

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """
        Compute the cosine similarity between two vectors.
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_relevant_file(self, query):
        """
        Identify the file most relevant to the given query based on embedding similarity.
        """
        if not self.index:
            return None

        try:
            query_emb = get_embedding(query)
        except Exception as e:
            print(f"[Index] Error getting query embedding: {e}")
            return None

        best_score = -1
        best_file = None

        for file_path, data in self.index.items():
            similarity = self.cosine_similarity(query_emb, data.get("embedding"))
            if similarity > best_score:
                best_score = similarity
                best_file = file_path

        print(f"[Index] Best match for query: {best_file} (similarity: {best_score:.3f})")
        return best_file

    def search_files_by_name(self, name_query):
        """
        Search for files whose names contain the given substring (case-insensitive).
        """
        matches = [file_path for file_path in self.index.keys() if name_query.lower() in file_path.name.lower()]
        if matches:
            print(f"[Index] Found {len(matches)} file(s) matching '{name_query}'.")
        else:
            print(f"[Index] No files found matching '{name_query}'.")
        return matches

    def select_file_from_matches(self, matches):
        """
        If multiple files match a query, let the user choose one.
        """
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        print("[Select] Multiple files found:")
        for idx, file in enumerate(matches):
            print(f"  {idx + 1}. {file}")
        while True:
            choice = input("Select file by number (or press Enter for the first match): ").strip()
            if not choice:
                return matches[0]
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(matches):
                    return matches[idx]
                else:
                    print("[Select] Invalid selection. Try again.")
            except ValueError:
                print("[Select] Please enter a valid number.")

    def aggregate_context(self, max_files=5, snippet_lines=9999999):
        """
        Aggregate context from up to N files for summarization.
        """
        sorted_files = sorted(self.index.items(), key=lambda item: len(item[1]["content"]), reverse=True)
        context_parts = []
        for file_path, data in sorted_files[:max_files]:
            lines = data["content"].splitlines()[:snippet_lines]
            snippet = "\n".join(lines)
            context_parts.append(f"File: {file_path}\n{snippet}\n")
        return "\n".join(context_parts)

    def update_framework_file(self, framework_file_path=FRAMEWORK_FILE):
        """
        Generate or update a JSON framework file that provides a quick reference
        for the codebase, including metadata and structure insights.
        """
        framework = {"files": []}

        for file_path, data in self.index.items():
            file_stats = os.stat(file_path)
            lines = data["content"].splitlines()
            file_size = file_stats.st_size  # File size in bytes
            last_modified = datetime.fromtimestamp(file_stats.st_mtime).isoformat()  # Last modified timestamp

            # Extract functions and class definitions
            functions = []
            classes = []
            imports = 0

            try:
                tree = ast.parse(data["content"])
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports += 1
            except SyntaxError:
                pass  # Skip if the file is not a valid Python script

            file_info = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "line_count": len(lines),
                "file_size": file_size,
                "last_modified": last_modified,
                "imports": imports,
                "functions": functions,
                "classes": classes,
                "snippet": "\n".join(lines[:20]) if lines else ""
            }
            framework["files"].append(file_info)

        try:
            with open(framework_file_path, "w", encoding="utf-8") as f:
                json.dump(framework, f, indent=4)
            print(f"[Framework] Updated framework file: {framework_file_path}")
        except Exception as e:
            print(f"[Framework] Failed to update framework file: {e}")

# --- File-System Watcher for Auto Refresh ---

class CodeIndexEventHandler(FileSystemEventHandler):
    def __init__(self, code_index: CodeIndex):
        self.code_index = code_index

    def on_modified(self, event):
        if not event.is_directory:
            print(f"[Watcher] Modified: {event.src_path}")
            self.code_index.refresh_index()

    def on_created(self, event):
        if not event.is_directory:
            print(f"[Watcher] Created: {event.src_path}")
            self.code_index.refresh_index()

    def on_deleted(self, event):
        if not event.is_directory:
            print(f"[Watcher] Deleted: {event.src_path}")
            self.code_index.refresh_index()

def start_file_watcher(code_index: CodeIndex):
    """
    Start a background thread to watch the project directory for changes.
    """
    if Observer is None:
        print("[Watcher] watchdog not available. Skipping auto refresh.")
        return

    event_handler = CodeIndexEventHandler(code_index)
    observer = Observer()
    observer.schedule(event_handler, path=str(code_index.project_root), recursive=True)
    observer_thread = threading.Thread(target=observer.start, daemon=True)
    observer_thread.start()
    print(f"[Watcher] Started file system watcher on {code_index.project_root}.")

# --- Diff and Change Engine ---

def apply_unified_diff(diff_text, file_path):
    """
    Apply a unified diff patch to a file.
    - Backs up the original file before applying the patch.
    - Uses `python-patch` to apply the changes.
    - Displays a preview of the differences before confirming the changes.
    - If the user rejects the patch, it restores from backup.
    """
    file_path = file_path if isinstance(file_path, str) else str(file_path)
    backup_path = file_path + ".bak"

    try:
        # Save a backup of the original file
        shutil.copy(file_path, backup_path)
        print(f"[Patch] Backup created: {backup_path}")

        # Read the original file content
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.readlines()

        # Ensure diff_text is in bytes for python-patch
        if isinstance(diff_text, str):
            diff_text = diff_text.encode("utf-8")

        # Apply patch using python-patch
        pset = patch.fromstring(diff_text)
        if not pset:
            raise ValueError("No valid patch found in diff output.")
        if not pset.apply(root=str(file_path.parent)):
            raise ValueError("Patch application failed.")

        print(f"[Patch] Patch successfully applied to {file_path}")

        # Read the new content after patching
        with open(file_path, "r", encoding="utf-8") as f:
            new_content = f.readlines()

        # Display differences using difflib
        diff = difflib.unified_diff(
            original_content, new_content,
            fromfile="Before Patch",
            tofile="After Patch",
            lineterm=""
        )
        diff_output = "\n".join(diff)
        print("[Patch] Differences preview:")
        print(diff_output)

        # Ask user to confirm changes
        confirm = input("Keep these changes? (y/n): ").strip().lower()
        if confirm != "y":
            # Restore backup if rejected
            shutil.copy(backup_path, file_path)
            print(f"[Patch] Changes reverted; original file restored from {backup_path}")
        else:
            # Remove backup if changes are confirmed
            try:
                os.remove(backup_path)
                print(f"[Patch] Changes confirmed; backup removed.")
            except OSError:
                print("[Patch] Failed to delete backup, but changes remain.")

    except Exception as e:
        print(f"[Patch] An error occurred: {e}")
        # Attempt to restore backup if an error occurs
        try:
            shutil.copy(backup_path, file_path)
            print(f"[Patch] Restored original file from backup due to error.")
        except Exception as ex:
            print(f"[Patch] Failed to restore backup: {ex}")

def log_change(message, file_path, log_file_path=CHANGE_LOG_FILE):
    """
    Log a change event with a description of the change.
    """
    entry = f"{time.asctime()}: {message} on {file_path}\n"
    try:
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(entry)
        print(f"[Log] Recorded change for {file_path}")
    except Exception as e:
        print(f"[Log] Failed to write change log: {e}")

def process_view_log():
    """
    Display the change log file to the user.
    """
    log_file = Path(CHANGE_LOG_FILE)
    if not log_file.exists():
        print("[View Log] No change log found.")
        return
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        print("\n[View Log] Change Log Contents:")
        print("-" * 40)
        print(content)
        print("-" * 40)
    except Exception as e:
        print(f"[View Log] Error reading change log: {e}")

# --- Dynamic Content Adjustor Functions ---

def extract_file_references(text):
    """
    Extract potential file references from text.
    Uses a regex to look for patterns like 'file:filename.ext' or 'file = filename.ext'.
    """
    pattern = r'file\s*[:=]\s*([\w\-.]+)'
    return re.findall(pattern, text, re.IGNORECASE)

def build_dynamic_context(new_instruction, code_index, conversation_history):
    """
    Combine the new instruction, recent conversation history, and codebase snippets
    into a dynamic context string.
    """
    # Summarize recent conversation history (prune if too long)
    pruned_history = prune_conversation_history(conversation_history)
    recent_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in pruned_history[-10:]
    )

    # Identify any mentioned files in the new instruction
    mentioned_files = extract_file_references(new_instruction)
    
    # For each referenced file, retrieve a snippet of its content
    file_contexts = []
    for file_ref in mentioned_files:
        matches = code_index.search_files_by_name(file_ref)
        selected_file = code_index.select_file_from_matches(matches)
        if selected_file:
            try:
                with open(selected_file, "r", encoding="utf-8") as f:
                    content = f.read()
                snippet = content[:500]  # First 500 characters as a snippet
                file_contexts.append(f"File: {selected_file}\nSnippet:\n{snippet}\n")
            except Exception as e:
                print(f"[Dynamic Context] Failed to read {selected_file}: {e}")

    dynamic_context = (
        f"Recent Conversation:\n{recent_history}\n\n"
        f"New Instruction:\n{new_instruction}\n\n"
        f"Relevant File Contexts:\n" + "\n".join(file_contexts)
    )
    return dynamic_context

def prune_conversation_history(history, max_entries=20):
    """
    If the conversation history grows too long, summarize older entries.
    This simple implementation keeps only the most recent `max_entries` messages.
    A more advanced implementation might perform summarization.
    """
    if len(history) > max_entries:
        # Here you could call a summarization API or algorithm.
        print(f"[History] Pruning conversation history (keeping last {max_entries} entries).")
        return history[-max_entries:]
    return history

# --- Chat Interface and Command Processing ---

def build_prompt(user_instruction, code_content, dynamic_context=""):
    """
    Build a conversation prompt for the chat API.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert code assistant integrated into an IDE. "
                "Analyze the provided code snippet along with the conversation context and user instruction, "
                "then produce a concise unified diff that implements the requested change. "
                "Output only the diff with minimal commentary."
            )
        },
        {
            "role": "user",
            "content": (
                f"{dynamic_context}\n\n"
                f"Instruction: {user_instruction}\n\n"
                f"Code:\n{code_content}\n\n"
                "Please provide the unified diff that implements this change."
            )
        }
    ]
    return messages

def process_chat_instruction(user_instruction, code_index: CodeIndex, conversation_history):
    """
    Process a modification request from the user, using dynamic context to determine the proper file.
    """
    print(f"[Process] Modification instruction: {user_instruction}")

    # First, check if the instruction explicitly references a file name.
    file_ref = None
    match = re.search(r'file\s*[:=]\s*([^\s]+)', user_instruction, re.IGNORECASE)
    if match:
        file_name = match.group(1)
        matches = code_index.search_files_by_name(file_name)
        file_ref = code_index.select_file_from_matches(matches)
        if file_ref:
            print(f"[Process] Using explicit file reference: {file_ref}")

    # If no explicit reference, try to use the dynamic context (chat history and embeddings)
    if not file_ref:
        file_ref = code_index.get_relevant_file(user_instruction)
        if file_ref:
            print(f"[Process] Selected file via embedding search: {file_ref}")

    if file_ref is None:
        print("[Process] No relevant file found for this instruction.")
        return

    # Get code content from the chosen file.
    try:
        with open(file_ref, "r", encoding="utf-8") as f:
            code_content = f.read()
    except Exception as e:
        print(f"[Process] Error reading {file_ref}: {e}")
        return

    # Build a dynamic context string.
    dynamic_context = build_dynamic_context(user_instruction, code_index, conversation_history)

    # Build the prompt combining dynamic context, instruction, and code content.
    messages = build_prompt(user_instruction, code_content, dynamic_context)
    # Optionally, include pruned conversation history.
    pruned_history = prune_conversation_history(conversation_history)
    messages = pruned_history + messages

    try:
        diff_output = generate_chat_response(messages)
        print("[Process] Proposed diff:\n", diff_output)
    except Exception as e:
        print(f"[Process] Error generating diff: {e}")
        return

    # Add the assistant's response to conversation history.
    conversation_history.append({"role": "assistant", "content": diff_output})

    user_approval = input("Apply this diff? (y/n): ")
    if user_approval.lower().startswith("y"):
        try:
            apply_unified_diff(diff_output, file_ref)
            code_index.update_file(file_ref)
            log_change(user_instruction, file_ref)
        except Exception as e:
            print(f"[Process] Failed to apply diff: {e}")
    else:
        print("[Process] Diff discarded by user.")

def process_explanation_request(code_index: CodeIndex):
    """
    Generate a high-level explanation of the code base.
    """
    context = code_index.aggregate_context()
    prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert code analyst. Given the aggregated context from a code base, "
                "provide a high-level summary describing the main idea, architecture, and purpose of the code."
            )
        },
        {
            "role": "user",
            "content": (
                "Here is some context from the code base:\n" + context +
                "\n\nPlease provide a concise summary of the overall idea and structure of this code base."
            )
        }
    ]
    try:
        explanation = generate_chat_response(prompt)
        print("[Explanation]\n", explanation)
    except Exception as e:
        print(f"[Explanation] Failed to generate explanation: {e}")

def process_traceback(traceback_text, code_index: CodeIndex):
    """
    Process an error traceback and generate a patch.
    """
    print("[Traceback] Received traceback for analysis.")
    file_match = re.search(r'File "([^"]+)"', traceback_text)
    if file_match:
        file_path_str = file_match.group(1)
        file_path = Path(file_path_str)
        if file_path.exists():
            print(f"[Traceback] Extracted file: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
            except Exception as e:
                print(f"[Traceback] Error reading {file_path}: {e}")
                return
        else:
            print(f"[Traceback] File {file_path} does not exist in the code base.")
            return
    else:
        print("[Traceback] No file extracted; using embedding search.")
        file_path = code_index.get_relevant_file(traceback_text)
        if file_path is None:
            print("[Traceback] Unable to locate a relevant file.")
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()
        except Exception as e:
            print(f"[Traceback] Error reading {file_path}: {e}")
            return

    messages = [
        {
            "role": "system",
            "content": (
                "You are a debugging assistant. Given the error traceback and associated code snippet, analyze the issue "
                "and provide a unified diff patch to fix the error. Output only the diff."
            )
        },
        {
            "role": "user",
            "content": (
                f"Traceback:\n{traceback_text}\n\n"
                f"Code from {file_path}:\n{code_content}\n\n"
                "Please provide a unified diff that fixes the issue."
            )
        }
    ]
    try:
        diff_output = generate_chat_response(messages)
        print("[Traceback] Proposed diff:\n", diff_output)
    except Exception as e:
        print(f"[Traceback] Failed to generate fix: {e}")
        return

    user_approval = input("Apply this diff? (y/n): ")
    if user_approval.lower().startswith("y"):
        try:
            apply_unified_diff(diff_output, file_path)
            code_index.update_file(file_path)
            log_change("Traceback fix", file_path)
        except Exception as e:
            print(f"[Traceback] Failed to apply diff: {e}")
    else:
        print("[Traceback] Diff discarded by user.")

def process_context_request(code_index: CodeIndex, batch_size: int = 30):
    """
    Retrieve file content or context by referencing file names or descriptions.
    Loads the full file in batches.
    """
    user_input = input("Enter file name or description to fetch context: ").strip()
    if not user_input:
        print("[Context] No input provided.")
        return

    matches = code_index.search_files_by_name(user_input)
    selected_file = code_index.select_file_from_matches(matches)
    if selected_file is None:
        selected_file = code_index.get_relevant_file(user_input)
        if selected_file:
            print(f"[Context] Using context from: {selected_file}")
        else:
            print("[Context] No file found based on the input.")
            return

    try:
        with open(selected_file, "r", encoding="utf-8") as f:
            file_lines = f.readlines()
    except Exception as e:
        print(f"[Context] Error reading {selected_file}: {e}")
        return
    
    total_lines = len(file_lines)
    print(f"\n[Context] Loading content from {selected_file} in batches of {batch_size} lines...")
    
    for start in range(0, total_lines, batch_size):
        end = min(start + batch_size, total_lines)
        snippet = "".join(file_lines[start:end])
        print(f"\n[Context] Batch {start + 1}-{end} of {total_lines} lines:\n{'-'*40}\n{snippet}\n{'-'*40}\n")
        user_continue = input("Load next batch? (y/n): ").strip().lower()
        if user_continue != 'y':
            break


# --- New Feature: AI Flows (Multi-step interactive session) ---

def process_flow(code_index: CodeIndex, conversation_history):
    print("[Flow] Starting AI Flow session. Type 'end' to finish.")
    while True:
        instruction = input("Flow Instruction: ").strip()
        if instruction.lower() == "end":
            print("[Flow] Ending AI Flow session.")
            break

        target_file = code_index.get_relevant_file(instruction)
        if not target_file:
            print("[DEBUG] No relevant file found for instruction.")
            continue

        filename = os.path.basename(target_file)

        # Ensure AI generates a strict unified diff patch
        system_prompt = (
            f"You are modifying the file '{filename}'. "
            "You MUST generate a valid unified diff patch. "
            "Your response must start with:\n"
            f"--- a/{filename}\n+++ b/{filename}\n"
            "Ensure that all necessary modifications are included. "
            "DO NOT add explanations or extra text—ONLY output the patch."
        )

        # Read the current file contents
        with open(target_file, "r", encoding="utf-8") as f:
            file_contents = f.read()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the current content of {filename}:\n\n{file_contents}"},
            {"role": "user", "content": instruction}
        ]

        # Generate the unified diff patch
        diff_patch = generate_chat_response(messages).strip()

        print(f"[DEBUG] Generated Diff Patch:\n{diff_patch}")

        # Validate if the AI-generated patch follows the correct format
        if not diff_patch.startswith(f"--- a/{filename}") or "+++" not in diff_patch:
            print("[ERROR] Invalid patch format detected.")
            print("[DEBUG] AI response:\n", diff_patch)
            continue

        # Create a backup before applying changes
        backup_path = str(target_file) + ".bak"
        shutil.copy(target_file, backup_path)

        # Apply the diff patch correctly
        try:
            apply_unified_diff(diff_patch, target_file)
            print("[Flow] Patch applied successfully.")
        except Exception as e:
            print(f"[Patch] Failed to apply: {e}")
            print("[Patch] Restoring original file from backup.")
            shutil.copy(backup_path, target_file)
            
# --- New Feature: Code Completion (AI Code Completion / Supercomplete) ---

def process_code_completion(code_index: CodeIndex):
    """
    Generate code completion suggestions using natural language prompts.
    """
    file_query = input("Enter file name for code completion: ").strip()
    if not file_query:
        print("[Complete] No file name provided.")
        return
    matches = code_index.search_files_by_name(file_query)
    selected_file = code_index.select_file_from_matches(matches)
    if not selected_file:
        print("[Complete] No file found.")
        return
    try:
        with open(selected_file, "r", encoding="utf-8") as f:
            code_content = f.read()
    except Exception as e:
        print(f"[Complete] Error reading {selected_file}: {e}")
        return
    context_line = input("Enter context (a few lines or a description where completion is needed): ").strip()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a code completion assistant. Given a partial code snippet and context, "
                "provide the best possible code completion. Output only the completed code snippet."
            )
        },
        {
            "role": "user",
            "content": (
                f"File: {selected_file}\nContext: {context_line}\n\n"
                f"Existing code:\n{code_content}\n\n"
                "Please provide a code completion suggestion."
            )
        }
    ]
    try:
        completion = generate_chat_response(messages)
        print("[Complete] Code Completion Suggestion:\n", completion)
    except Exception as e:
        print(f"[Complete] Failed to generate code completion: {e}")

# --- New Feature: Codebase Chat (Interactive Q&A about the code base) ---

def process_codebase_chat(code_index: CodeIndex):
    """
    Allow the user to ask natural language questions about the codebase.
    """
    question = input("Enter your codebase query: ").strip()
    if not question:
        print("[Chat] No question provided.")
        return
    context = code_index.aggregate_context()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful codebase chat assistant. Answer questions about the codebase clearly and concisely."
            )
        },
        {
            "role": "user",
            "content": f"Codebase context:\n{context}\n\nQuestion: {question}"
        }
    ]
    try:
        answer = generate_chat_response(messages)
        print("[Chat] Answer:\n", answer)
    except Exception as e:
        print(f"[Chat] Failed to generate response: {e}")

# --- Main Loop / Integration ---

def main():
    """
    Main entry point for the agentic IDE tool.
    """
    print("[Main] Welcome to the Agentic IDE Tool!")
    proj_input = input(f"Enter the project directory path [default: {DEFAULT_PROJECT_ROOT}]: ").strip()
    project_root = Path(proj_input) if proj_input else DEFAULT_PROJECT_ROOT
    if not project_root.exists() or not project_root.is_dir():
        print(f"[Main] The specified path '{project_root}' is not a valid directory. Exiting.")
        return

    # Build the code index.
    code_index = CodeIndex(project_root)
    code_index.build_index()

    # Start file system watcher if possible.
    start_file_watcher(code_index)

    print("\n[Main] Commands available:")
    print("  modify     - Modify code based on an instruction.")
    print("  explain    - Get a high-level explanation of the code base.")
    print("  traceback  - Provide an error traceback to generate a fix.")
    print("  context    - Retrieve file content or context by referencing file names or descriptions.")
    print("  flow       - Start an interactive multi-step modification flow.")
    print("  complete   - Get code completion suggestions based on context.")
    print("  chat       - Ask questions about the codebase in natural language.")
    print("  refresh    - Refresh the index (detect new/deleted files).")
    print("  clearcache - Clear the embedding cache.")
    print("  viewlog    - View the change log.")
    print("  exit       - Quit the tool.\n")

    conversation_history = []  # Conversation history includes both user and assistant messages.

    while True:
        command = input("Command (modify/explain/traceback/context/flow/complete/chat/refresh/clearcache/viewlog/exit): ").strip().lower()
        if command in ["exit", "quit"]:
            print("[Main] Exiting Agentic IDE Tool. Goodbye!")
            break
        elif command == "modify":
            instruction = input("Enter your modification instruction: ").strip()
            if instruction:
                process_chat_instruction(instruction, code_index, conversation_history)
                conversation_history.append({"role": "user", "content": instruction})
        elif command == "explain":
            process_explanation_request(code_index)
        elif command == "traceback":
            print("Paste the error traceback (end with an empty line):")
            lines = []
            while True:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            traceback_text = "\n".join(lines)
            if traceback_text:
                process_traceback(traceback_text, code_index)
        elif command == "context":
            process_context_request(code_index)
        elif command == "flow":
            process_flow(code_index, conversation_history)
        elif command == "complete":
            process_code_completion(code_index)
        elif command == "chat":
            process_codebase_chat(code_index)
        elif command == "refresh":
            code_index.refresh_index()
        elif command == "clearcache":
            clear_embedding_cache()
        elif command == "viewlog":
            process_view_log()
        else:
            print("[Main] Unrecognized command. Please choose from modify, explain, traceback, context, flow, complete, chat, refresh, clearcache, viewlog, or exit.")

if __name__ == "__main__":
    main()
