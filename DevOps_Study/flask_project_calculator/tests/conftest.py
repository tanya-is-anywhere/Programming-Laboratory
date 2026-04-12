import os
import tempfile
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flaskr import create_app

@pytest.fixture
def app():
    app = create_app({
        'TESTING': True,
    })
    yield app

@pytest.fixture
def client(app):
    with app.test_client() as client:
        yield client

def test_config():
    assert not create_app().testing
    assert create_app({'TESTING': True}).testing

def test_hello(client):
    response = client.get('/hello')
    assert response.data == b'Hello, World!'


@pytest.fixture
def runner(app):
    return app.test_cli_runner()

