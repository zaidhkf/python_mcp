import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from config.settings import UPLOAD_DIR, VECTOR_STORE_DIR
from rag_mcp.chunking import chunk_source
from rag_mcp.vector_store import INDEXES, load_source, search_source


def fake_embeddings(texts):
    vectors = []
    for text in texts:
        text = text.lower()
        vectors.append([1.0, 0.0] if any(term in text for term in ('launch', 'milestone', 'december'))
                       else [0.0, 1.0])
    return np.asarray(vectors, dtype='float32')


class SemanticRAGTests(unittest.TestCase):
    def tearDown(self):
        INDEXES.clear()

    def test_table_rows_are_traceable_chunks(self):
        chunks, metadata = chunk_source('titanic.csv')
        self.assertEqual(len(chunks), 891)
        self.assertEqual(metadata['rows'], 891)
        self.assertEqual(chunks[0]['row'], 1)
        self.assertIn('survived:', chunks[0]['text'])

    def test_semantic_index_and_threshold(self):
        name = f'rag_{uuid4().hex}.txt'
        path = UPLOAD_DIR / name
        test_store = VECTOR_STORE_DIR / f'test_{uuid4().hex}'
        test_store.mkdir()
        path.write_text('The project launch milestone is scheduled for December.', encoding='utf-8')
        try:
            with patch('rag_mcp.vector_store.VECTOR_STORE_DIR', test_store), \
                    patch('rag_mcp.vector_store.encode_texts', side_effect=fake_embeddings):
                loaded = load_source(name)
                relevant = search_source('When is the milestone?', name)
                unrelated = search_source('penguin habitat', name)
                INDEXES.clear()
                persisted = search_source('When is the launch?', name)
            self.assertEqual(loaded['indexed'], 1)
            self.assertEqual(relevant['results'][0]['filename'], name)
            self.assertIn('December', relevant['results'][0]['text'])
            self.assertEqual(unrelated['results'], [])
            self.assertIn('December', persisted['results'][0]['text'])
        finally:
            path.unlink(missing_ok=True)
            for stored_file in test_store.iterdir():
                stored_file.unlink()
            test_store.rmdir()


if __name__ == '__main__':
    unittest.main()
