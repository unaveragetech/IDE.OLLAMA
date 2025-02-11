---

# 🌟 IDE.OLLAMA 🌟
[📝 System Documentation](SYSTEM_DOCUMENTATION.md) | [📚 Further Reading](Details.md) | [📚 Download ollama windows](https://ollama.com/download/OllamaSetup.exe)

## Overview
IDE.OLLAMA is a community-driven project aimed at creating a lightweight, single-script Integrated Development Environment (IDE) for beginners and users who are tired of learning complex frameworks. The goal is to leverage Ollama and clever coding practices to maintain a powerful yet compact IDE for average users.

## 🎯 Project Goals
- 🗃️ Create a centralized, singular script (`ide.py`) serving as a lightweight IDE.
- 🛠️ Ensure all functionality is built into the `ide.py` script.
- ✨ Allow writing and maintaining other scripts and files at runtime, but include all standard functions within the main script.
- 🧹 Maintain clean, modular, and extendable code sections.

## 📜 Repository Rules
1. **Single Main Program**: The main program must be `ide.py`.
2. **Functionality**: All core functionality must be built into `ide.py`. Additional scripts and files can be created at runtime but should not contain standard functions.
3. **Modularity**: Each function should be semi-modular, extendable, or removable without breaking overall functionality.
4. **Clean Code**: Code must be clean and organized in sections denoted by spacers in the script.

## 🌟 Features
- **Configuration Management**: Load and manage configurations from a `config.json` file.
- **Embedding Cache**: Persistent in-memory cache for embeddings.
- **API Helper Functions**: Retrieve embeddings and generate chat responses using the Ollama API.
- **Code Indexing and Search**: Index codebase files, search by name, and find relevant files via embeddings.
- **File-System Watcher**: Automatically refresh the index upon file changes using `watchdog`.
- **Diff and Change Engine**: Apply unified diffs, log changes, and handle patch applications.
- **Dynamic Content Adjustment**: Extract file references and build dynamic context for chat interactions.
- **Interactive Commands**: Modify code, explain the codebase, handle tracebacks, retrieve context, start interactive flows, and more.

## 🚀 Getting Started
To get started with IDE.OLLAMA:
1. Ensure you have Ollama running.
2. Fork the repository.
3. Clone your fork and navigate to the project directory.

### Running the IDE
To run the IDE, execute the `ide.py` script:
```sh
python ide.py
```

Follow the on-screen instructions to use various commands and features.

## 🤝 Contributing
We welcome contributions to enhance IDE.OLLAMA. To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes, ensuring they adhere to the project rules.
4. Document all changes and the reasoning behind them.
5. Submit a Pull Request.

### Example Contribution
1. **Fork and Clone**:
    ```sh
    git clone https://github.com/your-username/IDE.OLLAMA.git
    cd IDE.OLLAMA
    ```
2. **Create a Branch**:
    ```sh
    git checkout -b feature/new-feature
    ```
3. **Make Changes**: Edit `ide.py` to add your feature.
4. **Commit and Push**:
    ```sh
    git add ide.py
    git commit -m "Add new feature to IDE"
    git push origin feature/new-feature
    ```
5. **Submit a Pull Request**: Go to your fork on GitHub and open a Pull Request to the main repository.

## 📜 License
This project is licensed under the Sduc License. See the `Sduc` file for details.

## 📞 Contact
For any questions or support, open an issue on the GitHub repository.

---

✨ Happy coding with IDE.OLLAMA! ✨
