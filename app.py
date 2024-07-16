import flask
from flask import request, render_template, g
import plotly.express as px
import plotly.graph_objects as go
from models_container.EstimatorsBTC import EstimatorsBTC
from graph_creator.graph_creator import GraphBTC


app = flask.Flask(__name__)


def get_engine():
    if 'engine' not in g:
        g.engine = EstimatorsBTC()
    return g.engine


def close_database(exception):
    engine = g.pop('engine', None)

    if engine is not None:
        engine.close()


@app.route("/")
def home():
    return render_template("mainpage.html")


@app.route("/performance/<period>/")
def performance(period):
    print("PERIOD", period)
    db = get_engine().modelDB
    performance_rf = db.get_model_performance("RandomForest")
    performance_ab = db.get_model_performance("AdaBoost")
    performance_gb = db.get_model_performance("GradientBoost")
    
    fig_rf = GraphBTC("Random Forest", performance_rf, [period]).get_graph()
    fig_ab = GraphBTC("Ada Boost", performance_ab, [period]).get_graph()
    fig_gb = GraphBTC("Gradient Boost", performance_gb, [period]).get_graph()

    return render_template("performance.html", rf_graph=fig_rf.to_json(), ab_graph=fig_ab.to_json(), gb_graph=fig_gb.to_json())


@app.route("/models/")
def models():
    engine = get_engine()
    pred = engine.get_prediction_today()
    
    return render_template("models.html",
                            gb_pred = pred["GradientBoost"],
                            ab_pred = pred["AdaBoost"],
                            rf_pred = pred["RandomForest"])


@app.route("/about/")
def about():
    return render_template("about.html")