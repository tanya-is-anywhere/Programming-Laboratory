import pytest


@pytest.mark.parametrize(('a', 'b', 'expected'), [
    ('20.1', '30.5', b'20.1 + 30.5 = 50.6'),
    ('10', '12', b'10.0 + 12.0 = 22.0'),
    ('test', '3', b'You need entry only numbers'),
    ('5', 'test', b'You need entry only numbers'),
    ('test', 'test', b'You need entry only numbers'),
])
def test_summa(client, a, b, expected):
    response = client.get(f'/summa/{a},{b}')
    assert expected in response.data


@pytest.mark.parametrize(('a', 'b', 'expected'), [
    ('21', '7', b'21.0 - 7.0 = 14.0'),
    ('10', '12', b'10.0 - 12.0 = -2.0'),
    ('30', '20', b'30.0 - 20.0 = 10.0'),
    ('test', '3', b'You need entry only numbers'),
    ('5', 'test', b'You need entry only numbers'),
    ('test', 'test', b'You need entry only numbers'),
])
def test_difference(client, a, b, expected):
    response = client.get(f'/difference/{a},{b}')
    assert expected in response.data


@pytest.mark.parametrize(('a', 'b', 'expected'), [
    ('5', '12', b'5.0 * 12.0 = 60.0'),
    ('10', '12', b'10.0 * 12.0 = 120.0'),
    ('2.5', '4', b'2.5 * 4.0 = 10.0'),
    ('test', '3', b'You need entry only numbers'),
    ('5', 'test', b'You need entry only numbers'),
    ('test', 'test', b'You need entry only numbers'),
])
def test_multiply(client, a, b, expected):
    response = client.get(f'/multiply/{a},{b}')
    assert expected in response.data


@pytest.mark.parametrize(('a', 'b', 'expected'), [
    ('2.5', '4', b'2.5 / 4.0 = 0.625'),
    ('10', '2', b'10.0 / 2.0 = 5.0'),
    ('10', '0', b'Error: division by zero'),
    ('test', '3', b'You need entry only numbers'),
    ('5', 'test', b'You need entry only numbers'),
    ('test', 'test', b'You need entry only numbers'),
])
def test_quotient(client, a, b, expected):
    response = client.get(f'/quotient/{a},{b}')
    assert expected in response.data