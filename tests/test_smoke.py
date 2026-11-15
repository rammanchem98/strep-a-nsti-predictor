from src import config


def test_config_features():
    assert len(config.FEATURES) == 8
    assert set(config.NUMERIC) | set(config.BINARY) == set(config.FEATURES)
