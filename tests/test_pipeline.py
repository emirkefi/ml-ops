from model_pipeline import prepare_data, train_model


def test_prepare_data():

    X_train, X_test, y_train, y_test = prepare_data()

    assert len(X_train) > 0
    assert len(X_test) > 0

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_train_model():

    X_train, X_test, y_train, y_test = prepare_data()

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")
