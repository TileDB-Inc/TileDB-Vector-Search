import json

import tiledb

from tiledb.vector_search.evals.runner import read_index_embedding_model_name


def test_read_index_embedding_model_name_from_sentence_transformer_key(tmp_path):
    uri = str(tmp_path / "idx")
    tiledb.group_create(uri)
    with tiledb.Group(uri, "w") as g:
        g.meta["sentence_transformer_model"] = "org/example-model"
    assert read_index_embedding_model_name(uri) == "org/example-model"


def test_read_index_embedding_model_name_prefers_sentence_transformer_key(tmp_path):
    uri = str(tmp_path / "idx")
    tiledb.group_create(uri)
    with tiledb.Group(uri, "w") as g:
        g.meta["sentence_transformer_model"] = "primary"
        g.meta["embedding_kwargs"] = json.dumps({"model_name_or_path": "ignored"})
    assert read_index_embedding_model_name(uri) == "primary"


def test_read_index_embedding_model_name_fallback_embedding_kwargs(tmp_path):
    uri = str(tmp_path / "idx")
    tiledb.group_create(uri)
    with tiledb.Group(uri, "w") as g:
        g.meta["embedding_kwargs"] = json.dumps(
            {"model_name_or_path": "fallback/from-kwargs"}
        )
    assert read_index_embedding_model_name(uri) == "fallback/from-kwargs"


def test_read_index_embedding_model_name_missing_returns_none(tmp_path):
    uri = str(tmp_path / "idx")
    tiledb.group_create(uri)
    with tiledb.Group(uri, "w") as g:
        g.meta["embedding_kwargs"] = json.dumps({})
    assert read_index_embedding_model_name(uri) is None
