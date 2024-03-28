from tiledb.vector_search.object_readers import DirectoryTextReader
from tiledb.vector_search.object_readers.directory_reader import match_uri


def test_match_uri():
    # Matching patterns
    assert match_uri(
        search_uri="s3://dir",
        file_uri="s3://dir/subdir/A.pdf",
        include="*.pdf",
    )
    assert match_uri(
        search_uri="s3://dir",
        file_uri="s3://dir/subdir/A.pdf",
        include="*/*.pdf",
    )
    assert match_uri(
        search_uri="s3://dir",
        file_uri="s3://dir/subdir/A.pdf",
        include="subdir/*.pdf",
    )
    assert match_uri(
        search_uri="s3://dir",
        file_uri="s3://dir/subdir/A.pdf",
        include="*/A*",
    )
    assert match_uri(
        search_uri="s3://dir",
        file_uri="s3://dir/subdir/A.pdf",
        suffixes=[".png", ".pdf"],
    )

    # Not matching patterns
    assert not match_uri(
        search_uri="s3://dir/", file_uri="s3://dir/subdir/A.pdf", include="*.png"
    )
    assert not match_uri(
        search_uri="s3://dir/",
        file_uri="s3://dir/subdir/A.pdf",
        include="*.pdf",
        exclude=["*A.pdf"],
    )
    assert not match_uri(
        search_uri="s3://dir/",
        file_uri="s3://dir/subdir/A.pdf",
        exclude=["*.png", "*.pdf"],
    )
    assert not match_uri(
        search_uri="s3://dir",
        file_uri="s3://dir/subdir/A.pdf",
        suffixes=[".png"],
    )


def test_list(tmpdir):
    dir = tmpdir.mkdir("dir")
    subdir1 = dir.mkdir("subdir1")
    subdir2 = dir.mkdir("subdir2")
    subsubdir = subdir2.mkdir("sub")

    a = subdir1.join("A.txt")
    a.write("content")
    a1 = subdir1.join("A1.pdf")
    a1.write("content")
    b = subdir1.join("B.doc")
    b.write("content")
    c = subdir1.join("C.png")
    c.write("content")

    d = subdir2.join("D.txt")
    d.write("content")
    e = subdir2.join("E.txt")
    e.write("content")

    f = subsubdir.join("F.pdf")
    f.write("content")
    g = subsubdir.join("G.txt")
    g.write("content")
    reader = DirectoryTextReader(
        search_uri=str(dir), include="*[A|C|F]*", suffixes=[".pdf", ".png"]
    )
    reader.list_paths()
    assert "file://" + str(a) not in reader.paths
    assert "file://" + str(a1) in reader.paths
    assert "file://" + str(b) not in reader.paths
    assert "file://" + str(c) in reader.paths
    assert "file://" + str(d) not in reader.paths
    assert "file://" + str(e) not in reader.paths
    assert "file://" + str(f) in reader.paths
    assert "file://" + str(g) not in reader.paths
