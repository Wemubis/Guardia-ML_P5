# MachineLearning_P5


### 1. Analyse des types de transactions :

À partir de la distribution des types de transactions pour les transactions frauduleuses, il semble que les types 'TRANSFER' et 'CASH_OUT' soient plus courants dans les activités frauduleuses.

```python
plt.figure(figsize=(12, 6))
sns.countplot(x='type', data=df[df['isFraud'] == 1])
plt.title('Distribution of Transaction Types for Fraudulent Transactions')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

<br>

### 2. Analyse des montants :

La distribution des montants de transactions pour les transactions frauduleuses montre une variété de montants, certaines transactions ayant des valeurs significativement plus élevées.

```python
plt.figure(figsize=(12, 6))
sns.histplot(df[df['isFraud'] == 1]['amount'], bins=30, kde=True)
plt.title('Distribution of Transaction Amounts for Fraudulent Transactions')
plt.xlabel('Transaction Amount')
plt.ylabel('Count')
plt.show()
```

<br>

### 3. Analyse des soldes :

La distribution des soldes dans les transactions frauduleuses présente des motifs variés, indiquant des caractéristiques potentielles pour la détection de la fraude.

```python
balance_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
plt.figure(figsize=(15, 8))
for col in balance_cols:
    sns.histplot(df[df['isFraud'] == 1][col], bins=30, kde=True, label=col)
plt.title('Distribution of Balances for Fraudulent Transactions')
plt.xlabel('Balance')
plt.ylabel('Count')
plt.legend()
plt.show()
```

<br>

### 4. Analyse des tendances temporelles :

Le graphique de l'occurrence de la fraude au fil du temps ne révèle pas de tendance temporelle claire, mais il est essentiel de prendre en compte les caractéristiques liées au temps.

```python
plt.figure(figsize=(15, 6))
sns.lineplot(x='step', y='isFraud', data=df, ci=None)
plt.title('Fraud Occurrence Over Time')
plt.xlabel('Time Step')
plt.ylabel('Fraud Occurrence')
plt.show()
```

<br>

### 5. Analyse des transactions signalées :

Le nombre de transactions frauduleuses signalées parmi les transactions frauduleuses montre que la grande majorité des transactions frauduleuses ne sont pas signalées.

```python
plt.figure(figsize=(8, 6))
sns.countplot(x='isFlaggedFraud', hue='isFraud', data=df[df['isFraud'] == 1])
plt.title('Count of Flagged Fraud Transactions among Fraudulent Transactions')
plt.xlabel('isFlaggedFraud')
plt.ylabel('Count')
plt.show()
```

<br>

### Aperçu global :

- Le jeu de données est fortement déséquilibré, avec un pourcentage très faible de transactions étiquetées comme frauduleuses.
	
- Les types de transactions 'TRANSFER' et 'CASH_OUT' sont plus fréquents dans les activités frauduleuses.
	
- Les transactions frauduleuses impliquent souvent des montants plus élevés par rapport aux transactions non frauduleuses.
	
- Les soldes dans les comptes d'origine et de destination présentent des motifs variés pour les transactions frauduleuses.
	
- Il n'y a pas de tendance temporelle claire dans l'occurrence de la fraude, mais les caractéristiques liées au temps pourraient être pertinentes.
	
- Les transactions signalées sont rares parmi les transactions frauduleuses, indiquant des limites potentielles dans le mécanisme actuel de signalement.
