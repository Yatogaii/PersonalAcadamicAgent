# Design

## Collection Schema
We have 8 rows:
1. id: Primary ID, auto increment.
2. doc_id: Paper's unique id.
3. title: Title of paper.
4. abstract: Absctract of paper.
5. url: download url for paper's pdf(if avaliable).
6. content: One chunk of paper's content.
7. chunk_id: chunk id for paper's content.
8. vector: embedding vector for **Title + Absctract + chunks[chunk_id]**

Only `id`, `doc_id`, `title`, `abstract` is NOT nullable. We insert enteties with content lazily.

## Workflow
First when agent found a paper, it will insert `(id, doc_id, title, abstract, url)` to database.

When necessary, agent will call tools to download pdf and split it to chunks.
Then it will embedding the mixed content and insert them to database as an new entity.

> [!note]
> Vector Database is not good for update, it's always advised to insert instead of update.