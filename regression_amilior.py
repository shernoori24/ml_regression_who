import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
# 6. ENTRAÎNEMENT DE PLUSIEURS MODÈLES
# ============================================

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
}

results = {}

print("\n Entraînement et évaluation des modèles :")
print("=" * 60)

for name, model in models.items():
    # Entraînement
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Cross-validation (sur un échantillon pour gagner du temps)
    cv_sample = min(10000, len(X_train))
    cv_scores = cross_val_score(
        model, X_train.iloc[:cv_sample], y_train.iloc[:cv_sample],
        cv=5, scoring='r2', n_jobs=-1
    )
    
    results[name] = {
        'model': model,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"\n{name} :")
    print(f"  R² Score       : {r2:.4f}")
    print(f"  MAE            : {mae:.4f}%")
    print(f"  RMSE           : {rmse:.4f}%")
    print(f"  CV R² (mean±std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================
# 7. ANALYSE DES FEATURES IMPORTANTES
# ============================================

best_model_name = max(results, key=lambda k: results[k]['r2'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Meilleur modèle : {best_model_name}")

# Extraire l'importance des features
if hasattr(best_model, 'feature_importances_'):
    # Pour Random Forest
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
elif hasattr(best_model, 'coef_'):
    # Pour les modèles linéaires
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': best_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False).head(15)

print("\n Top 15 des features les plus importantes :")
print(importance.to_string(index=False))

# ============================================
# 8. VISUALISATIONS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Comparaison des modèles
ax1 = axes[0, 0]
model_names = list(results.keys())
r2_scores = [results[m]['r2'] for m in model_names]
colors = ['green' if r2 == max(r2_scores) else 'steelblue' for r2 in r2_scores]

ax1.barh(model_names, r2_scores, color=colors)
ax1.set_xlabel('R² Score')
ax1.set_title('Comparaison des Performances des Modèles')
ax1.grid(axis='x', alpha=0.3)

for i, v in enumerate(r2_scores):
    ax1.text(v + 0.01, i, f'{v:.4f}', va='center')

# 2. Features importantes
ax2 = axes[0, 1]
if 'Importance' in importance.columns:
    ax2.barh(importance['Feature'].head(10), importance['Importance'].head(10))
    ax2.set_xlabel('Importance')
else:
    ax2.barh(importance['Feature'].head(10), importance['Coefficient'].head(10).abs())
    ax2.set_xlabel('|Coefficient|')
ax2.set_title(f'Top 10 Features - {best_model_name}')
ax2.grid(axis='x', alpha=0.3)

# 3. Prédictions vs Réalité
ax3 = axes[1, 0]
y_pred_best = results[best_model_name]['predictions']
ax3.scatter(y_test, y_pred_best, alpha=0.3, s=10)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Prédiction parfaite')
ax3.set_xlabel('Taux de mortalité réel (%)')
ax3.set_ylabel('Taux de mortalité prédit (%)')
ax3.set_title(f'Prédictions vs Réalité ({best_model_name})')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Distribution des erreurs
ax4 = axes[1, 1]
errors = y_test - y_pred_best
ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Erreur de prédiction (%)')
ax4.set_ylabel('Fréquence')
ax4.set_title('Distribution des Erreurs de Prédiction')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n Analyse terminée !")