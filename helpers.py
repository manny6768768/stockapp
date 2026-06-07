import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import numpy as np
from pickle import load

class AttentionPooling(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)
    def call(self, x):
        scores  = tf.squeeze(x @ self.W, -1)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.einsum("bt,btf->bf", weights, x)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

mag_model = load_model("models/qqq_lstm_magnitude.keras",
                       custom_objects={"AttentionPooling": AttentionPooling})
dir_model_3d  = load(open("models/qqq_xgb_dir_3d.pkl",  "rb"))
dir_model_5d  = load(open("models/qqq_xgb_dir_5d.pkl",  "rb"))
dir_model_10d = load(open("models/qqq_xgb_dir_10d.pkl", "rb"))
scaler_mag = load(open("models/qqq_scaler_mag.pkl", "rb"))
scaler_dir = load(open("models/qqq_scaler_dir.pkl", "rb"))

# Trigger Keras graph tracing/compilation now (at boot) instead of on the first
# live request, where it can blow past gunicorn's worker timeout and get killed.
mag_model.predict(np.zeros((1, 60, scaler_mag.n_features_in_), dtype=np.float32), verbose=0)

MAG_MEAN = 0.01313
MAG_STD  = 0.00890 

def add_rsi(df, window=14):
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()
    df['Rsi'] = 100 - (100 / (1 + avg_gain / avg_loss))
    return df

def preprocess_data(df, look_back=60):
    d = df.copy()
    d['Return']        = d['Close'].pct_change()
    d['Return_3d']     = d['Close'].pct_change(3)
    d['Return_5d']     = d['Close'].pct_change(5)
    d['Return_10d']    = d['Close'].pct_change(10)
    d['Return_20d']    = d['Close'].pct_change(20)
    d['Vol_5']         = d['Return'].rolling(5).std()
    d['Vol_20']        = d['Return'].rolling(20).std()
    d['Vol_Ratio']     = d['Vol_5'] / d['Vol_20']
    d['Rsi']           = add_rsi(d)['Rsi']
    d['Rsi_Change']    = d['Rsi'].diff()
    d['Return_Vol']    = d['Return']    * d['Vol_20']
    d['Momentum_Vol']  = d['Return_5d'] * d['Vol_20']
    exp12 = d['Close'].ewm(span=12).mean()
    exp26 = d['Close'].ewm(span=26).mean()
    d['MACD']          = exp12 - exp26
    d['MACD_Signal']   = d['MACD'].ewm(span=9).mean()
    d['MACD_Hist']     = d['MACD'] - d['MACD_Signal']
    ma20  = d['Close'].rolling(20).mean()
    std20 = d['Close'].rolling(20).std()
    d['BB_Position']   = (d['Close'] - (ma20 - 2*std20)) / (4*std20 + 1e-10)
    d['VIX_Return']    = d['VIX'].pct_change()
    d['VIX_MA20']      = d['VIX'].rolling(20).mean()
    d['VIX_Ratio']     = d['VIX'] / (d['VIX_MA20'] + 1e-10)
    d['VIX_Vol']       = d['VIX'].rolling(5).std()
    d['SMA_50']        = d['Close'].rolling(50).mean()
    d['SMA_200']       = d['Close'].rolling(200).mean()
    d['SMA_cross']     = d['SMA_50'] / (d['SMA_200'] + 1e-10) - 1
    d['Price_vs_SMA50']= d['Close']  / (d['SMA_50']  + 1e-10) - 1

    d.dropna(inplace=True)

    features_mag = [
        'Return',
        'Vol_5', 'Vol_20', 'Vol_Ratio',
        'Return_Vol', 'Momentum_Vol',
        'VIX_Return', 'VIX_MA20', 'VIX_Ratio',
        'VIX', 'VIX_Vol',
    ]
    features_dir = [
        'Return',
        'Return_3d', 'Return_5d', 'Return_10d', 'Return_20d',
        'Rsi', 'Rsi_Change',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Position',
        'SMA_cross', 'Price_vs_SMA50',
        'VIX_Return', 'VIX_Ratio',
    ]

    d = d[list(set(features_mag + features_dir + ['Date']))]

    last = len(d) - 1
    X_mag = d[features_mag].iloc[last-look_back+1:last+1].values[np.newaxis, :, :]
    X_dir = d[features_dir].iloc[last-look_back+1:last+1].values[np.newaxis, :, :]
    dates = np.array([d['Date'].iloc[last]])

    n, lb, nf_mag = X_mag.shape
    n, lb, nf_dir = X_dir.shape
    X_mag = scaler_mag.transform(X_mag.reshape(-1, nf_mag)).reshape(n, lb, nf_mag)
    X_dir = scaler_dir.transform(X_dir.reshape(-1, nf_dir)).reshape(n, lb, nf_dir)

    return X_mag, X_dir, dates

def predict(X_mag, X_dir):
    raw = mag_model.predict(X_mag, verbose=0)          # (n, 3)
    last_dir = X_dir[:, -1, :]                         # (n, nf_dir)

    pred_vol_3d  = raw[:, 0] * MAG_STD + MAG_MEAN
    pred_vol_5d  = raw[:, 1] * MAG_STD + MAG_MEAN
    pred_vol_10d = raw[:, 2] * MAG_STD + MAG_MEAN

    pred_move_3d  = pred_vol_3d  * np.sqrt(3)  * np.sqrt(2 / np.pi)
    pred_move_5d  = pred_vol_5d  * np.sqrt(5)  * np.sqrt(2 / np.pi)
    pred_move_10d = pred_vol_10d * np.sqrt(10) * np.sqrt(2 / np.pi)

    prob_3d  = dir_model_3d.predict_proba(last_dir)[:, 1]
    prob_5d  = dir_model_5d.predict_proba(last_dir)[:, 1]
    prob_10d = dir_model_10d.predict_proba(last_dir)[:, 1]

    return {
        "move_3d":  pred_move_3d,
        "move_5d":  pred_move_5d,
        "move_10d": pred_move_10d,
        "prob_3d":  prob_3d,
        "prob_5d":  prob_5d,
        "prob_10d": prob_10d,
    }
