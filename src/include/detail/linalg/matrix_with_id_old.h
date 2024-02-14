
class MatrixWithId {
  MatrixWithId(const Matrix& matrix, const vector<id_type>& ids)
      : matrix_(matrix)
      , ids_(ids) {
  }
  // no copy constructor
  // have a move constructor / operator
  Matrix& matrix() {
    return matrix_;
  }
  Matrix& matrix() {
    return ids_;
  }

 private:
  Matrix<feature_type, etc.> matrix_;
  vector<id_type> ids_;
}
