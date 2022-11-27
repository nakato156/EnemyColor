import pandas as pd
import numpy as np
import cv2 as cv
import joblib
from PIL import ImageColor
from importlib.util import find_spec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class EnemyColor():
    def __init__(self, default_load=True):
        self._model_rgb = {}
        self._path_colors = "data/colors.csv"
        self._path_models = "data/models"
        if default_load: self.load_models()

    def clasif_data_color(self, color)->dict:
        rgb = {"r":[], "g":[], "b":[]}
        for r,g,b in color:
            rgb["r"].append(r)
            rgb["g"].append(g)
            rgb["b"].append(b)
        return {k:np.array(v) for k,v in rgb.items()}
    
    def load_data_csv(self, path_csv:str = "", is_hex=True):
        df = pd.read_csv(path_csv.strip() if path_csv.strip()  else self._path_colors)
        if is_hex:
            df["color"]= df["color"].apply(lambda x: ImageColor.getcolor(f"#{x}", "RGB"))
            df["contrast"] = df["contrast"].apply(lambda x: ImageColor.getcolor(f"#{x}", "RGB"))

        self.rgb_1 = self.clasif_data_color(df.color.to_list())
        self.rgb_2 = self.clasif_data_color(df.contrast.to_list())
        self.loaded = True
        return self

    def load_models(self):
        self._model_rgb = {i: joblib.load(f'{self._path_models}/modelo_{i}.pkl') for i in "rgb"}
        return self

    def train(self, score=.25, save:bool=False)->dict:
        for i in "rgb":
            while True:
                X_train, X_test, y_train, y_test = train_test_split(self.rgb_1[i], self.rgb_2[i], train_size=.9)
                modelo = LinearRegression()
                modelo.fit(X = X_train.reshape(-1, 1), y = y_train)
                model_score = modelo.score(X_train.reshape(-1, 1), y_train)
                if model_score>score: break
                else: print(f"score: {model_score}")
            if save: joblib.dump(modelo, f'{self._path_models}/modelo_{i}.pkl')
            self._model_rgb[i] = modelo
        return self._model_rgb

    def predict(self, predict_color:np.array, preview:bool=True)->tuple:
        """
        :predict_color: array de 3 elementos correspondiete a los canales rgb
        Ejemplo:
            predict(np.array([140, 250, 200]))
        :preview: booleana que indica si se mostrará una previsualizacion del color
        :return: una tupla de los canales rgb del color de contraste predicho
        """
        predict_rgb = {}
        for i, c in enumerate("rgb"):
            modelo = self._model_rgb[c]
            predicciones = modelo.predict(X = predict_color[i].reshape(-1, 1))
            predict_rgb[c] = [int(x) for x in predicciones[0:3,]]
        self.rgb = tuple(x[0] for x in predict_rgb.values())
        if preview: self.preview(*self.rgb)
        return self.rgb
    
    def preview(self, *rgb):
        if not rgb: rgb=self.rgb
        prev_img = np.zeros((100,100,3),np.uint8)
        for x in range(100):
            for y in range(100):
                prev_img[x,y] = [*rgb]
        name = "MyImage"
        cv.imwrite(f"{name}.png", prev_img)
        
        if not find_spec("matplotlib.pyplot") or find_spec("matplotlib.image"):
            import matplotlib.pyplot as plt
            import matplotlib.image as img

        image = img.imread(f'{name}.png')
        plt.imshow(image)
        plt.show()