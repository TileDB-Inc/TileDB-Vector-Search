from tiledb.vector_search import _tiledbvspy as vspy

index_group_uri = "/private/var/folders/jb/5gq49wh97wn0j7hj6zfn9pzh0000gn/T/pytest-of-parismorgan/pytest-347/test_vamana_index0/array"
parts_array_uri = "file:///private/var/folders/jb/5gq49wh97wn0j7hj6zfn9pzh0000gn/T/pytest-of-parismorgan/pytest-347/test_vamana_index0/array/shuffled_vectors"

ctx = vspy.Ctx({})

def test_foo():
  data = vspy.FeatureVectorArray(ctx, parts_array_uri)
  print('data.feature_type_string()', data.feature_type_string())
  print('Done.')