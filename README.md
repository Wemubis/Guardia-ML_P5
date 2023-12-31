# MachineLearning_P5

## Informations sur le jeu de données

- **Colonnes :**
  - `step` : Heure de la transaction sur un mois.
  - `type` : Type de transaction.
  - `amount` : Montant de la transaction.
  - `oldbalanceOrg` : Solde d'origine dans le compte d'origine.
  - `newbalanceOrig` : Nouveau solde dans le compte d'origine.
  - `oldbalanceDest` : Solde d'origine dans le compte de destination.
  - `newbalanceDest` : Nouveau solde dans le compte de destination.
  - `isFraud` : Booléen indiquant si la transaction est frauduleuse (1) ou non (0).
  - `isFlaggedFraud` : Booléen indiquant si la transaction est signalée comme frauduleuse (1) ou non (0).

<br><br>

## Contenu du dépôt

### 1. Exploration du jeu de données

- Affichage d'informations de base sur le jeu de données avec `df.info()` :

    ```python
    print(df.info())
    ```

- Présentation des statistiques sommaires pour les colonnes numériques avec `df.describe()` :

    ```python
    print(df.describe())
    ```

- Présentation des premières lignes du jeu de données avec `df.head()` :

    ```python
    print(df.head())
    ```

<br>

### 2. Analyse de la colonne 'isFraud'

- Exploration de la colonne 'isFraud' pour comprendre la distribution des transactions frauduleuses et non frauduleuses :

    ```python
    # Affichage des valeurs uniques dans la colonne 'isFraud'
    print(df['isFraud'].unique())

    # Affichage de la distribution des valeurs dans la colonne 'isFraud'
    print(df['isFraud'].value_counts())

    # Affichage du pourcentage de transactions frauduleuses/non frauduleuses
    print(df['isFraud'].value_counts(normalize=True) * 100)

    # Affichage des valeurs manquantes dans le jeu de données
    print(df.isnull().sum())
    ```

<br>

### 3. Analyse approfondie des données

- Réalisation d'une analyse détaillée de certaines caractéristiques :

  - Types de transactions : Analyse de la distribution des types de transactions pour les transactions frauduleuses :

    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    sns.countplot(x='type', data=df[df['isFraud'] == 1])
    plt.title('Distribution of Transaction Types for Fraudulent Transactions')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    ```

  - Analyse des montants : Exploration de la distribution des montants de transactions pour les transactions frauduleuses :

    ```python
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['isFraud'] == 1]['amount'], bins=30, kde=True)
    plt.title('Distribution of Transaction Amounts for Fraudulent Transactions')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Count')
    plt.show()
    ```

  - Boîte à moustaches des montants de transactions par statut de fraude :

    ```python
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='isFraud', y='amount', data=df)
    plt.title('Box Plot of Transaction Amounts by Fraud Status')
    plt.xlabel('isFraud')
    plt.ylabel('Amount')
    plt.show()
    ```

  - Graphique en barres du pourcentage de transactions frauduleuses/non frauduleuses :

    ```python
    plt.figure(figsize=(8, 6))
    fraud_percentage.plot(kind='bar', color=['green', 'red'])
    plt.title('Percentage of Fraud/Non-Fraud Transactions')
    plt.xlabel('isFraud')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.show()
    ```

  - Matrice de corrélation :

    ```python
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    ```
<br>

### Aperçu global :
	
- Les transactions 'TRANSFER' et 'CASH_OUT' dominent les fraudes, confirmant leur rôle prépondérant dans ces incidents.

- L'analyse des soldes d'origine et de destination révèle une diversité de motifs pour les fraudes, soulignant l'utilité potentielle de ces informations.

- Aucune tendance temporelle claire n'émerge dans la fraude, mais des caractéristiques temporelles méritent une exploration plus approfondie.

- Les transactions signalées restent peu fréquentes dans les cas de fraudes, suggérant des lacunes dans le processus de signalement.

- La forte asymétrie entre transactions frauduleuses et non frauduleuses (99.87% contre 0.13%) souligne le défi du déséquilibre de classe dans l'entraînement des modèles.

