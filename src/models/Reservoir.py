import matplotlib.pyplot as plt
import reservoirpy as rpy
import numpy as np
from itertools import product


class ESN:

    def __init__(self, n_units: int, n_input: int, wanted_sr: float = None, lr: float = 1, W_in_type: str = "uniform",
                 W_in_scale: float = 0.1, W_type: str = "uniform", W_scale: float = 1, W_degree: int = 6,
                 seed_W_in: int = None, seed_W: int = None, ridge: float = 0, activation: str = "tanh"):
        """
        Initialiseur de la classe custom ESN, basée sur ReservoirPy.
        :param n_units: int. Nombre d'unités dans le réservoir.
        :param n_input: int. Dimension de l'entrée (nombre de features ou de séries).
        :param wanted_sr: float. Rayon spectral de la matrice interne du réservoir. Essentiellement, valeur propre
        absolue maximale de la matrice. Sert èa avoir un état d'echo intéressant pour le chaos. Typiquement légèrement
        au-dessus de 1, mais dépend du problème. Défaut à `None`, on laisse la matrice comme elle est.
        :param lr: float. Leaking rate (taux de fuite) du modèle de réservoir. Défaut à 1.
        :param W_in_type: str. Type de distribution pour la matrice W_in. Défaut à "uniform", seule possibilité pour
        l'instant.
        :param W_in_scale: float. Facteur d'échelle pour la génération de la matrice W_in. Défaut à 0.1. Dans le cas
        uniforme, cela correspond à des entrées distribuées dans l'intervalle uniforme [-0.1, 0.1].
        :param W_type: str. Type de distribution pour la matrice W. Défaut à "uniform", seule possibilité pour
        l'instant.
        :param W_scale: float. Facteur d'échelle pour la génération de la matrice W. Défaut à 1. Dans le cas uniforme,
        cela correspond à des entrées non nulle distribuées dans l'intervalle uniforme [-1, 1].
        :param W_degree: float. Degré de connectivité de la matrice W. Essentiellement, une proportion
        W_degree / n_units entrées seront non nulles. Défaut à 6.
        :param seed_W_in: int. Seed pour la génération aléatoire de W_in.
        :param seed_W: Seed pour la génération aléatoire de W.
        :param ridge: float. Paramètre pour la régularisation Ridge.
        :param activation: str. Fonction d'activation. Défaut à "tanh", tangente hyperbolique.
        """
        self.n_units = n_units
        self.n_input = n_input
        self.seed_W_in = seed_W_in
        self.seed_W = seed_W
        self.wanted_sr = wanted_sr
        self.lr = lr
        self.W_in_type = W_in_type
        self.W_in_scale = W_in_scale
        self.W_in = self.__generate_W_in()
        self.W_type = W_type
        self.W_scale = W_scale
        self.W_degree = W_degree
        self.W = self.__generate_W()
        if self.wanted_sr is not None:
            self.__rescale_sr()
        self.ridge = ridge
        self.Ridge = rpy.nodes.Ridge(ridge=self.ridge)
        self.activation = activation
        self.Reservoir = rpy.nodes.Reservoir(Win=self.W_in, W=self.W, lr=self.lr, activation=self.activation)
        self.ESN = self.Reservoir >> self.Ridge

    def __generate_W_in(self):
        """
        Méthode privée pour générer W_in.
        :return: W_in, un ndarray de NumPy, de shape (n_units, n_input).
        """
        if self.W_in_type != "uniform":
            raise ValueError(f"Génération de W_in selon `{self.W_in_type}` inconnue.")
        rng = np.random.default_rng(self.seed_W_in)
        W_in = rng.uniform(-self.W_in_scale, self.W_in_scale, size=(self.n_units, self.n_input))
        return W_in

    def __generate_W(self):
        """
        Méthode privée pour générer W.
        :return: W, un ndarray de NumPy, de shape (n_units, n_units).
        """
        if self.W_type != "uniform":
            raise ValueError(f"Génération de W_in selon `{self.W_type}` inconnue.")
        rng = np.random.default_rng(self.seed_W)
        p = self.W_degree / self.n_units
        mask = rng.random((self.n_units, self.n_units)) < p
        W = rng.uniform(-self.W_scale, self.W_scale, size=(self.n_units, self.n_units)) * mask
        return W

    def __rescale_sr(self):
        """
        Méthode privée permettant de transformer W de sorte que la matrice aie une valeur propre absolue maximale
        spécifique
        :return: Rien.
        """
        rho = self.spectral_radius
        if rho != 0:
            self.W *= self.wanted_sr / rho

    @property
    def spectral_radius(self):
        """
        Propriété permettant de calculer la valeur propre absolue maximale de W.
        :return:
        """
        return float(np.max(np.abs(np.linalg.eigvals(self.W))))

    def fit(self, X: np.ndarray, Y: np.ndarray, skip: int = 0):
        """
        Méthode permettant d'entraîner le Ridge.
        :param X: np.ndarray. Données d'entraînement.
        :param Y: np.ndarray. Données cibles à la prédiction.
        :param skip: int. Nombre de valeurs de X et Y à passer au début.
        :return: le retour de fit de ReservoirPy.
        """
        return self.ESN.fit(X, Y, warmup=skip)

    def autonomous_prediction(self, start: np.ndarray, n: int):
        """
        Méthode permettant d'effectuer la prédiction autonome d'un état de départ, normalement le dernier état de la
        série d'entraînement.
        :param start: np.ndarray. État de départ pour la prédiction.
        :param n: int. Nombre d'éléments à prédire.
        :return: preds, un ndarray de NumPy, contenant les prédictions.
        """
        preds = []
        x = start
        for _ in range(n):
            x = self.ESN(x).squeeze()
            preds.append(x)
        return np.array(preds)

    def predict(self, start: np.ndarray, n: int):
        return self.autonomous_prediction(start, n)

    def __call__(self, *args, **kwargs):
        return self.autonomous_prediction(*args, **kwargs)

    @staticmethod
    def prediction_error(predictions: np.ndarray, true: np.ndarray, dimensionwise: bool = False):
        """
        Méthode utilitaire permettant de calculer l'erreur de prédiction utilisant la RMSE (root mean squared error).
        :param predictions: np.ndarray. Valeurs prédites.
        :param true: np.ndarray. Vraies valeurs.
        :param dimensionwise: bool. Booléen spécifiant si on veut une valeur par dimension (feature). Défaut à True.
        :return:
        """
        rmse = rpy.observables.rmse(true, predictions, dimensionwise)
        return rmse


class GridOptim:
    # TODO: Faire une métrique custom, RMSE mais normalisée pour laisser plus d'importance au début des prédictions
    # TODO: Genre, une exponentielle décroissante.
    def __init__(self, args_vals: dict, *model_args, **model_kwargs):
        self.args_vals = args_vals
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def _make_models(self) -> dict:
        keys = self.args_vals.keys()
        values = self.args_vals.values()
        all_combs = product(*values)
        models = dict()
        for c in all_combs:
            current_kwargs = dict(zip(keys, c))
            m = ESN(*self.model_args, **current_kwargs, **self.model_kwargs)
            models[m] = current_kwargs
        return models

    def optimize(self, X_fit, y_fit, predict_start, y_predict, metric: callable, fit_args: tuple = (),
                 predict_args: tuple = (), metric_args: tuple = ()) -> dict:
        min_err = np.inf
        best_param = None
        models = self._make_models()
        for m in models.keys():
            m.fit(X_fit, y_fit, *fit_args)
            n = len(y_predict)
            predictions = m.predict(predict_start, n, *predict_args)
            err = metric(predictions, y_predict, *metric_args)
            if err < min_err:
                best_param = models[m]
                min_err = err
        return best_param
