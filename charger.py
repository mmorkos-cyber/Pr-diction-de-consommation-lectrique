import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Chargement du fichier
df = pd.read_csv("tp1-pedagogique-conso.csv", parse_dates=["date_heure"])

# Découverte du jeu de données
print("Dimensions :", df.shape)
print("\nAperçu :")
print(df.head())

print("\nInformations :")
print(df.info())

print("\nStatistiques :")
print(df.describe())

print("\nValeurs manquantes :")
print(df.isna().sum())

df.set_index("date_heure")["consommation_mw"].plot(figsize=(12, 4), title="Consommation dans le temps")
plt.show()
 
df.groupby("jour_semaine")["consommation_mw"].mean().plot(kind="bar", title="Moyenne par jour de semaine")
plt.show()
 
df.groupby("heure")["consommation_mw"].mean().plot(kind="bar", title="Moyenne par heure")
plt.show()
 
df.plot.scatter(x="temperature", y="consommation_mw", alpha=0.3, title="Consommation vs temperature")
plt.show()

colonnes_numeriques = ["temperature", "jour_semaine", "heure", "est_weekend", "mois", "consommation_mw"]
matrice_corr = df[colonnes_numeriques].corr()
 
plt.figure(figsize=(8, 6))
sns.heatmap(matrice_corr, annot=True, cmap="coolwarm", center=0)
plt.title("Matrice de correlation")
plt.show()

def mois_vers_saison(mois):
    if mois in [12, 1, 2]:
        return "hiver"
    elif mois in [3, 4, 5]:
        return "printemps"
    elif mois in [6, 7, 8]:
        return "ete"
    else:
        return "automne"
 
df["saison"] = df["mois"].apply(mois_vers_saison)
df_saison = pd.get_dummies(df["saison"], prefix="saison")
df = pd.concat([df, df_saison], axis=1)
 
features = ["temperature", "jour_semaine", "heure", "est_weekend"] + list(df_saison.columns)
X = df[features]
y = df["consommation_mw"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
 
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

model_lineaire = LinearRegression()
model_lineaire.fit(X_train, y_train)
pred_lineaire = model_lineaire.predict(X_test)
 
mae_lineaire = mean_absolute_error(y_test, pred_lineaire)
print(f"MAE regression lineaire : {mae_lineaire:.0f} MW")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAE : {mae:.0f} MW - MAPE : {mape*100:.1f}%")

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
importances.plot(kind="bar", title="Importance des features")
plt.show()
 
residus = y_test.values - y_pred
 
plt.figure(figsize=(12, 4))
plt.plot(residus)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residus (erreur) dans le temps")
plt.show()
 
df_test = df.iloc[X_test.index].copy()
df_test["erreur_abs"] = abs(residus)
df_test.groupby("heure")["erreur_abs"].mean().plot(kind="bar", title="Erreur moyenne par heure")
plt.show()

baseline_pred = df["consommation_mw"].shift(24).iloc[X_test.index]
mae_baseline = mean_absolute_error(y_test, baseline_pred.bfill())
print(f"MAE baseline (J-1) : {mae_baseline:.0f} MW")