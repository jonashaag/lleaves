import lightgbm
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import lleaves

MODEL_DIRS_NUMERICAL = [
    "tests/models/boston_housing/",
    "tests/models/NYC_taxi/",
    "tests/models/single_tree/",
    "tests/models/tiniest_single_tree/",
]


MODEL_DIRS_CATEGORICAL = [
    "tests/models/mixed_categorical/",
    "tests/models/pure_categorical/",
    "tests/models/airline/",
]
CAT_BITVEC_CATEGORICAL = [
    (True, True, True, False, False),
    (True, True, True),
    (True, True, True, True, False, False),
]


@pytest.fixture(scope="session", params=MODEL_DIRS_NUMERICAL)
def llvm_lgbm_model(request):
    path = request.param
    llvm = lleaves.Model(model_file=path + "model.txt")
    llvm.compile()
    return (
        llvm,
        lightgbm.Booster(model_file=path + "model.txt"),
    )


@pytest.fixture(
    scope="session", params=zip(MODEL_DIRS_CATEGORICAL, CAT_BITVEC_CATEGORICAL)
)
def llvm_lgbm_model_cat(request):
    path, bitvec = request.param

    llvm = lleaves.Model(model_file=path + "model.txt")
    llvm.compile()
    return (
        llvm,
        lightgbm.Booster(model_file=path + "model.txt"),
        bitvec,
    )


def test_attribute_similarity(llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    assert llvm_model.num_feature() == lightgbm_model.num_feature()


@settings(deadline=1000)
@given(data=st.data())
def test_forest_llvm_mode(data, llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    input_data = data.draw(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            max_size=llvm_model.num_feature(),
            min_size=llvm_model.num_feature(),
        )
    )
    input_data = np.array([input_data])
    assert llvm_model.predict(input_data) == lightgbm_model.predict(input_data)


@settings(max_examples=10)
@given(data=st.data())
def test_batchmode(data, llvm_lgbm_model):
    llvm_model, lightgbm_model = llvm_lgbm_model
    input_data = []
    for i in range(10):
        input_data.append(
            data.draw(
                st.lists(
                    st.floats(allow_nan=True, allow_infinity=True),
                    max_size=llvm_model.num_feature(),
                    min_size=llvm_model.num_feature(),
                )
            )
        )
    input_data = np.array(input_data)
    np.testing.assert_array_equal(
        llvm_model.predict(input_data), lightgbm_model.predict(input_data)
    )


@given(data=st.data())
@settings(deadline=None)  # the airline model takes a few seconds to compile
def test_forest_llvm_mode_cat(data, llvm_lgbm_model_cat):
    llvm_model, lgbm_model, cat_bitvec = llvm_lgbm_model_cat

    input_cats = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=2 ** 31 - 2),
            min_size=sum(cat_bitvec),
            max_size=sum(cat_bitvec),
        )
    )
    input_floats = data.draw(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True),
            max_size=lgbm_model.num_feature() - sum(cat_bitvec),
            min_size=lgbm_model.num_feature() - sum(cat_bitvec),
        )
    )
    input_data = [
        input_cats.pop() if is_cat else input_floats.pop() for is_cat in cat_bitvec
    ]
    np.testing.assert_array_almost_equal(
        llvm_model.predict([input_data]), lgbm_model.predict([input_data]), decimal=15
    )
