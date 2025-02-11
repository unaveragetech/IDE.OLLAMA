# Quickstart and Explanation

## Overview

IDE.OLLAMA is designed to be a lightweight, single-script Integrated Development Environment (IDE) that simplifies coding and minimizes the learning curve associated with complex frameworks. This document provides a detailed explanation of how the `ide.py` script works and its key components.

## Key Components

### 1. Configuration Management
The script loads configuration settings from a `config.json` file if available. Default settings are defined in the `DEFAULT_CONFIG` dictionary.

### 2. Persistent Embedding Cache
An in-memory cache is used to store embeddings, which are synced to disk. Functions to load, save, and clear the cache are provided.

### 3. API Helper Functions
Functions like `get_embedding` and `generate_chat_response` interact with the Ollama API to retrieve embeddings and generate chat responses.

### 4. Code Indexing and Search
The `CodeIndex` class manages an in-memory index of the codebase, mapping file paths to their content and embedding vectors. It includes methods to build, refresh, and search the index.

### 5. File-System Watcher
The script uses `watchdog` to automatically refresh the code index when files are modified, created, or deleted.

### 6. Diff and Change Engine
Functions like `apply_unified_diff` and `log_change` handle the application of unified diff patches and logging changes.

### 7. Dynamic Content Adjustment
Functions to extract file references from text and build dynamic context strings are provided to enhance chat interactions.

### 8. Interactive Commands
The main loop of the script provides several commands for modifying code, explaining the codebase, handling tracebacks, retrieving context, starting interactive flows, generating code completions, and chatting about the codebase.

## Running the IDE

To run the IDE, execute the `ide.py` script:
```sh
python ide.py
```

### Commands Available
- `modify`: Modify code based on an instruction.
- `explain`: Get a high-level explanation of the code base.
- `traceback`: Provide an error traceback to generate a fix.
- `context`: Retrieve file content or context by referencing file names or descriptions.
- `flow`: Start an interactive multi-step modification flow.
- `complete`: Get code completion suggestions based on context.
- `chat`: Ask questions about the codebase in natural language.
- `refresh`: Refresh the index (detect new/deleted files).
- `clearcache`: Clear the embedding cache.
- `viewlog`: View the change log.
- `exit`: Quit the tool.

## How It Works

### Configuration Management
Configuration settings are loaded from `config.json` if it exists, otherwise, default settings are used.

### Embedding Cache
The embedding cache is loaded from `embedding_cache.json` and saved back to it after updates. The cache stores embeddings to minimize redundant API calls.

### API Interactions
The script interacts with the Ollama API to fetch embeddings for text and generate chat responses. It includes retry mechanisms for robustness.

### Code Indexing
The `CodeIndex` class builds an index of the project directory, storing the content and embeddings of files. It can refresh the index to reflect changes in the file system.

### File-System Watching
Using `watchdog`, the script watches for file changes and automatically refreshes the index.

### Applying Diffs
The script can apply unified diff patches to files, backing up the original file and allowing user confirmation before applying changes.

### Dynamic Context
When processing chat instructions, the script builds a dynamic context string that includes recent conversation history, the new instruction, and relevant file snippets.

### Command Processing
The main loop processes user commands, interacting with the code index and embedding cache to perform various tasks like modifying code, explaining the codebase, handling tracebacks, and more.

### Detailed Command Interactions

#### `modify`
- **Purpose**: Modify code based on an instruction.
- **Interaction**: Prompts the user for a modification instruction, processes the instruction to determine the relevant file, generates a unified diff, and applies the diff if approved by the user.

#### `explain`
- **Purpose**: Get a high-level explanation of the code base.
- **Interaction**: Aggregates context from the codebase and generates a summary explanation using the Ollama API.

#### `traceback`
- **Purpose**: Provide an error traceback to generate a fix.
- **Interaction**: Prompts the user to paste an error traceback, processes the traceback to identify the relevant file, and generates a patch to fix the error using the Ollama API.

#### `context`
- **Purpose**: Retrieve file content or context by referencing file names or descriptions.
- **Interaction**: Prompts the user for a file name or description, searches for the relevant file, and displays the content in batches.

#### `flow`
- **Purpose**: Start an interactive multi-step modification flow.
- **Interaction**: Initiates an AI-assisted flow session, allowing the user to provide instructions iteratively. Generates and applies unified diff patches based on user instructions.

#### `complete`
- **Purpose**: Get code completion suggestions based on context.
- **Interaction**: Prompts the user for a file name and context, and generates code completion suggestions using the Ollama API.

#### `chat`
- **Purpose**: Ask questions about the codebase in natural language.
- **Interaction**: Allows the user to ask questions about the codebase, generates responses based on aggregated context from the codebase using the Ollama API.

#### `refresh`
- **Purpose**: Refresh the index (detect new/deleted files).
- **Interaction**: Scans the project directory to update the code index with new or deleted files.

#### `clearcache`
- **Purpose**: Clear the embedding cache.
- **Interaction**: Clears the in-memory and on-disk embedding cache.

#### `viewlog`
- **Purpose**: View the change log.
- **Interaction**: Displays the contents of the change log file.

## Framework File
The `framework.json` file describes the structure of the codebase, including metadata and insights. When files are indexed, their content and embeddings are stored in this file for quick reference.

## Embeddings Information
Embeddings are vector representations of text used to capture semantic meaning. The script uses the Ollama API to generate embeddings for code snippets and other text. These embeddings are cached to minimize redundant API calls and are used for tasks like finding relevant files and generating code modifications.

To see more embedding-related content, visit [this search result](https://github.com/unaveragetech/IDE.OLLAMA/search?q=embedding).

