SAT GRADER — BOOK MODE INTEGRATION

Replace these files in the existing repository:
- app.py
- templates/index.html
- requirements.txt
- render.yaml
- rag/__init__.py
- rag/build_index.py
- rag/retriever.py

Keep your existing login.html and other project files.

BOOK FILES
The repository must contain the extracted book JSON files in a lowercase
books/ directory, for example:
  books/book1_extracted.json
  books/book2_extracted.json

The PDFs are not needed at runtime. Do not put chunks.json in books/.
The Render build command installs dependencies and regenerates rag/chunks.json.

LOCAL TEST
From the project root, run:
  pip install -r requirements.txt
  python -m rag.build_index
  python -m rag.retriever
  python app.py

DEPLOY
Commit and push the changed files and the extracted JSON files to the private
GitHub repository. Render will run the updated build command automatically.
Check the Render build logs for "Index preparation completed!" and the total
chunk count before testing Book Mode.

BEHAVIOR
- Standard Mode: existing single Claude answer call per image; no book retrieval.
- Book Mode: one Claude query-extraction call, BM25 top-five retrieval, then one
  Claude answer call. The answer response includes unique book/page references.
- Extended Thinking applies to the final answer call. Query extraction is a
  short standard Claude call.
