from rent_predictor import *
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
import math
from scipy.stats import norm


def test_rent():
    score = score_rent()
    assert score > 0.53
