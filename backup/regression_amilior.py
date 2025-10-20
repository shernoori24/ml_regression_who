import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import datetime

# ============================================
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ============================================

# Charger les données (garder TOUTES les données)
df = pd.read_csv('data/raw/full_clean_covid_19.csv')
locations = pd.read_csv('data/raw/full_clean_locations.csv')

# Merge avec les informations géographiques
df = df.merge(
    locations[['country_region', 'province_state', 'city', 'population', 'who_region']],
    left_on=['country', 'province_state', 'city'],
    right_on=['country_region', 'province_state', 'city'],
    how='left'
)

# Nettoyer
df = df.drop(columns=['country', 'province_state', 'city'], errors='ignore')
df = df.dropna(subset=['population', 'who_region'])

# ============================================
# 2. CRÉATION DE FEATURES AMÉLIORÉES
# ============================================

# Convertir la date
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Features temporelles
df['day'] = df['observation_date'].dt.day
df['month'] = df['observation_date'].dt.month
df['year'] = df['observation_date'].dt.year
df['day_of_week'] = df['observation_date'].dt.dayofweek
df['week_of_year'] = df['observation_date'].dt.isocalendar().week

# Features épidémiologiques AVANT de supprimer les colonnes
df['case_fatality_rate'] = (df['total_deaths'] / df['total_cases'] * 100).replace([np.inf, -np.inf], np.nan)
df['daily_case_rate'] = (df['new_cases'] / df['population'] * 100000).replace([np.inf, -np.inf], np.nan)
df['cumulative_case_rate'] = (df['total_cases'] / df['population'] * 100000).replace([np.inf, -np.inf], np.nan)

# Supprimer les lignes avec des valeurs manquantes ou infinies dans la cible
df = df[df['case_fatality_rate'].notna()]
df = df[np.isfinite(df['case_fatality_rate'])]

#  IMPORTANT : Ne pas utiliser total_recovered et active_cases car ils dépendent des décès
df = df.drop(columns=['observation_date', 'total_deaths', 'total_cases', 
                       'total_recovered', 'active_cases'], errors='ignore')

# ============================================
# 3. ÉCHANTILLONNAGE STRATIFIÉ (si nécessaire)
# ============================================

# Si le dataset est trop grand, échantillonner de manière stratifiée
if len(df) > 500000:
    # Stratifier par région et par quartile de mortalité
    df['mortality_quartile'] = pd.qcut(df['case_fatality_rate'], q=4, labels=False)
    df_sample = df.groupby(['who_region', 'mortality_quartile'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), 20000), random_state=42)
    )
    df = df_sample.drop(columns=['mortality_quartile'])
    print(f" Échantillonné à {len(df)} lignes de manière stratifiée")

# ============================================
# 4. ENCODAGE ET NORMALISATION
# ============================================

# Séparer features et cible
y = df['case_fatality_rate']
X = df.drop(columns=['case_fatality_rate'])

# Encodage des variables catégorielles
X = pd.get_dummies(X, drop_first=True)

# Remplir les valeurs manquantes
X = X.fillna(X.median())

# Standardisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# ============================================
# 5. SÉPARATION TRAIN/TEST
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n Taille du dataset :")
print(f"   Train : {len(X_train)} lignes")
print(f"   Test  : {len(X_test)} lignes")
print(f"   Features : {X_train.shape[1]}")

# ============================================
# 6. ENTRAÎNEMENT DU MODÈLE RANDOM FOREST
# ============================================

# Entraînement du modèle Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

print("\n Entraînement du modèle Random Forest :")
print("=" * 60)

# Entraînement
rf_model.fit(X_train, y_train)

# Prédictions
y_pred = rf_model.predict(X_test)

# Métriques
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Cross-validation (sur un échantillon pour gagner du temps)
cv_sample = min(10000, len(X_train))
cv_scores = cross_val_score(
    rf_model, X_train.iloc[:cv_sample], y_train.iloc[:cv_sample],
    cv=5, scoring='r2', n_jobs=-1
)

print(f"\nRandom Forest :")
print(f"  R² Score       : {r2:.4f}")
print(f"  MAE            : {mae:.4f}%")
print(f"  RMSE           : {rmse:.4f}%")
print(f"  CV R² (mean±std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================
# 7. ANALYSE DES FEATURES IMPORTANTES
# ============================================

print(f"\n🏆 Modèle Random Forest")

# Extraire l'importance des features
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

print("\n Top 15 des features les plus importantes :")
print(importance.to_string(index=False))

# ============================================
# 8. VISUALISATIONS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Performance du modèle Random Forest
ax1 = axes[0, 0]
metrics = ['R²', 'MAE', 'RMSE', 'CV R²']
values = [r2, mae, rmse, cv_scores.mean()]
colors = ['green', 'orange', 'red', 'blue']

ax1.bar(metrics, values, color=colors, alpha=0.7)
ax1.set_ylabel('Score')
ax1.set_title('Métriques de Performance - Random Forest')
ax1.grid(axis='y', alpha=0.3)

for i, v in enumerate(values):
    ax1.text(i, v + max(values)*0.01, f'{v:.4f}', ha='center')

# 2. Features importantes
ax2 = axes[0, 1]
ax2.barh(importance['Feature'].head(10), importance['Importance'].head(10))
ax2.set_xlabel('Importance')
ax2.set_title(f'Top 10 Features - Random Forest')
ax2.grid(axis='x', alpha=0.3)

# 3. Prédictions vs Réalité
ax3 = axes[1, 0]
ax3.scatter(y_test, y_pred, alpha=0.3, s=10)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Prédiction parfaite')
ax3.set_xlabel('Taux de mortalité réel (%)')
ax3.set_ylabel('Taux de mortalité prédit (%)')
ax3.set_title(f'Prédictions vs Réalité (Random Forest)')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Distribution des erreurs
ax4 = axes[1, 1]
errors = y_test - y_pred
ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Erreur de prédiction (%)')
ax4.set_ylabel('Fréquence')
ax4.set_title('Distribution des Erreurs de Prédiction')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 9. SÉRIALISATION DU MODÈLE
# ============================================

print("\n Sauvegarde du modèle...")
print("=" * 40)

# Créer le dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Préparer les métadonnées du modèle
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_info = {
    'model': rf_model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'performance': {
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    },
    'training_date': pd.Timestamp.now(),
    'model_type': 'RandomForestRegressor',
    'model_params': rf_model.get_params(),
    'dataset_size': {
        'train': len(X_train),
        'test': len(X_test),
        'features': X_train.shape[1]
    }
}

# Sauvegarder le modèle complet avec métadonnées
joblib.dump(model_info, f'models/complete_model_{timestamp}.pkl', compress=3)

# Sauvegarder aussi les composants séparément pour faciliter l'usage
joblib.dump(rf_model, 'models/random_forest_covid.pkl', compress=3)
joblib.dump(scaler, 'models/scaler_covid.pkl', compress=3)
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

# Afficher les informations de sauvegarde
print(" Modèle sauvegardé avec succès dans le dossier 'models/'")
print(f"    Modèle complet : complete_model_{timestamp}.pkl ({os.path.getsize(f'models/complete_model_{timestamp}.pkl')/1024:.1f} KB)")
print(f"    Modèle seul    : random_forest_covid.pkl ({os.path.getsize('models/random_forest_covid.pkl')/1024:.1f} KB)")
print(f"    Scaler        : scaler_covid.pkl ({os.path.getsize('models/scaler_covid.pkl')/1024:.1f} KB)")
print(f"    Features      : feature_names.pkl ({os.path.getsize('models/feature_names.pkl')/1024:.1f} KB)")

print("\n Pour charger le modèle plus tard :")
print("   import joblib")
print("   model_info = joblib.load('models/complete_model_[timestamp].pkl')")
print("   model = model_info['model']")
print("   scaler = model_info['scaler']")

print("\n Analyse et sauvegarde terminées !")