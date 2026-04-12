import os
from flask import Flask
def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/summa/<a>,<b>')
    def summa(a, b):
        try:
            num1 = float(a)
            num2 = float(b)
            return f"{num1} + {num2} = {num1+num2}"
        except ValueError:
            return "You need entry only numbers", 400

    @app.route('/difference/<a>,<b>')
    def difference(a, b):
        try:
            num1 = float(a)
            num2 = float(b)
            return f"{num1} - {num2} = {num1 - num2}"
        except ValueError:
            return "You need entry only numbers", 400

    @app.route('/multiply/<a>,<b>')
    def multiply(a, b):
        try:
            num1 = float(a)
            num2 = float(b)
            return f"{num1} * {num2} = {num1 * num2}"
        except ValueError:
            return "You need entry only numbers", 400

    @app.route('/quotient/<a>,<b>')
    def quotient(a, b):
        try:
            num1 = float(a)
            num2 = float(b)
            return f"{num1} / {num2} = {num1 / num2}"
        except ValueError:
            return "You need entry only numbers", 400
        except ZeroDivisionError:
            return "Error: division by zero", 400



    return app

# Для вызова использовать flask --app flaskr run --debug в папке проекта
