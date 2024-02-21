## On Type Erasure

Type erasure is accomplished as a three-layer cake:

- A non-templated abstract base class with pure virtual functions for the member functions that we want to expose from the typed C++ classes.
- An implementation class that inherits from the abstract base class. It is templated on the concrete class to be wrapped and keeps a member variable that is an object of the class to be wrapped. Its member functions (all of which are overrides of the abstract base class member functions) forward to the underlying C++ implementation by invoking the appropriate member. At this point, they internal data that is stored by the typed member variable also needs to be converted to the appropriate type before invoking the member variable function.
- A non-templated class that presents the user API. It internally defines the abstract base class and the implementation class. It has a `std::unique_ptr` to the abstract base class as a member variable. During construction (either by passing in already constructed vectors or by reading the index from a TileDB group, the appropriate template types for the internal data to be stored by the internal implementation are inferred and an object of the implementation class is constructed and stored in the `std::unique_ptr`.

To illustrate the basic idea, consider `FeatureVector`. In abbreviated form, where we just show a single function 'data', looks like this:

````c++
class FeatureVector {
    FeatureVector(const tiledb::Context& ctx, const std::string uri) {
      // get type of vector stored in uri array -- say, float
      feature_vector_ = std::make_unique<vector_impl<tdbVector<float>>>(ctx, uri);
    }
    auto data() {
     return feature_vector_->data();
   }

   class Base {
    virtual void* data() = 0;
  };

  template <class T>
  class Impl {
     explicit Impl(T&& t)
        : impl_vector_(std::forward<T>(t)) {
    }
    T impl_vector_;
  };

  std::unique<Base> feature_vector_;
};

The constructor to read the `FeatureVector` from a TileDB array has the following prototype:
```c++
FeatureVector(const tiledb::Context& ctx, const std::string& uri);
````

When that constructor is invoked, it first reads the schema associated with the `uri` and creates an implementation object based on that type. For example, if the type read from the schema (`feature_type`) is one of a `float` or `uint8`, the constructor dispatches like this:

```c++
switch (feature_type_) {
      case TILEDB_FLOAT32:
        vector_ = std::make_unique<vector_impl<tdbVector<float>>>(ctx, uri);
        break;
      case TILEDB_UINT8:
        vector_ = std::make_unique<vector_impl<tdbVector<uint8_t>>>(ctx, uri);
        break;
}
```

At this point, we have created a `std::unique_ptr` of the abstract base class that points to an object of the derived class.
If we invoke the `data` member function of the outer (type-erased) `FeatureVector` class, we dispatch to the corresponding member of the object stored in the `std::unique_ptr`:

```c++
 auto data() const {
    return feature_vector_->data();
  }
```

Since `feature_vector_` actually points to the derived implementation class, its `data` member function is then invoked:

```c++
void* data() override {
      return impl_vector_->data();
    }
```

We return a `void*` since `data()` is an override of the non-templated `Base` class.
(TODO: In a future PR maybe we can cast to an appropriate type extracted from the type of `vector_`?)
