# adabi_ml_studentibocciati
Progetto di Machine Learning di Laura Nuzzaci, obiettivo predire voto e promozione degli studenti.
## Project Overview

Il progetto ha l'obiettivo di prevedere il rendimento scolastico di uno studente a partire dal suo background familiare e dal suo stile di vita , dunque non dai voti già ottenuti durante l'anno. Vengono affrontati due problemi collegati sullo stesso dataset:

1. Prevedere il voto finale  dello studente (regressione)
2. Prevedere se lo studente sarà promosso o bocciato (classificazione)

L'idea di fondo è capire quanto fattori come l'istruzione dei genitori, le assenze, le bocciature pregresse, il consumo di alcol o le uscite con gli amici siano in grado di spiegare il rendimento scolastico, isolandoli deliberatamente dai voti intermedi (che sarebbero un predittore "troppo facile" e poco informativo sul resto).

## Dataset

- Fonte: [Kaggle — Student Performance Data](https://www.kaggle.com/datasets/devansodariya/student-performance-data/data)
- Dimensione: 395 studenti, 33 variabili, **nessun valore mancante**
- Target:
  - `G3` — voto finale (scala 0-20), usato come target di regressione
  - `G3 >= 10` — variabile binaria promosso/bocciato, derivata da `G3`, usata come target di classificazione

### Variabili principali

| Variabile | Significato |

| `Medu`, `Fedu` | Istruzione della madre e del padre (0 = nessuna, 4 = superiore) |
| `failures` | Numero di bocciature pregresse (0-3) |
| `goout` | Frequenza delle uscite con gli amici |
| `Dalc`, `Walc` | Consumo di alcol, rispettivamente infrasettimanale e nel weekend (1-5) |
| `absences` | Numero di assenze (0-75) |
| `G1`, `G2`, `G3` | Voti del primo periodo, secondo periodo e finale (0-20) |

`G1` e `G2` sono state escluse dalle feature usate per l'addestramento: essendo fortemente correlate con `G3`, il loro utilizzo avrebbe "nascosto" l'effetto reale delle altre variabili, che è invece l'oggetto dello studio. Il loro impatto viene comunque misurato separatamente, come controprova, nella sezione finale dei risultati.

Il 67,1% degli studenti risulta promosso e dunque le classi per la classificazione sono sbilanciate, un aspetto che ha condizionato diverse scelte successive nel progetto.

## Workflow

1. Data loading: caricamento del CSV, ispezione con `.head()`, `.info()`, `.describe()`
2. Data exploration: distribuzione del voto finale, boxplot di G3 rispetto a istruzione dei genitori e consumo di alcol, matrice di correlazione, relazione tra bocciature e voto medio
3. Data cleaning: nessun valore mancante rilevato — non è stata necessaria alcuna pulizia dei dati (vi è comunque la cella dedicata)
4. Preprocessing**: pipeline con `ColumnTransformer` — imputazione (mediana per le variabili numeriche, valore più frequente per quelle testuali), standardizzazione (`StandardScaler`) e codifica delle variabili categoriche (`OneHotEncoder`);
rimozione di `G1`, `G2`, `G3` dalle feature; creazione dei due target (`y_reg`, `y_clf`)
5. Split: train/test 80/20, stratificato sulla classe promosso/bocciato (316 righe di training, 79 di test)
6. Addestramento e confronto modelli: cross-validation a 5 fold, sia per la regressione sia per la classificazione
7. Tuning: ottimizzazione degli iperparametri della Random Forest di classificazione tramite `GridSearchCV`
8. Aggiustamento della soglia di decisione: per correggere l'effetto delle classi sbilanciate sul recall della classe minoritaria
9. Valutazione finale: sul test set, con classification report, matrici di confusione, RMSE
10. Analisi di interpretabilità: feature importance dei modelli Random Forest, per regressione e classificazione
11. Controprova: confronto dell'RMSE con e senza `G1`/`G2` per quantificare quanto i voti intermedi siano predittivi

## Models Used

| Task | Modelli confrontati | Modello scelto |

| Regressione (voto G3) | Regressione Lineare, Random Forest Regressor | Random Forest |
| Classificazione (promosso/bocciato) | Regressione Logistica, Random Forest Classifier | Regressione Logistica (in cross-validation) → Random Forest ottimizzata via `GridSearchCV` per tuning e interpretazione |

La scelta della Random Forest per il tuning, nonostante la Regressione Logistica avesse un'accuratezza in cross-validation leggermente superiore (67,7% contro 67,1%, differenza non significativa rispetto alla deviazione standard di entrambe), è motivata dal fatto che la Random Forest fornisce nativamente la feature importance, utile per l'obiettivo interpretativo del progetto.

## Results

### Regressione: previsione del voto finale (G3)

| Modello | RMSE medio (cross-validation) |
|---|---:|
| Regressione Lineare | 4.35 |
| Random Forest | 3.85 |

| Metric | Score |
|---|---:|
| RMSE sul test set | 3.91 |
| Deviazione standard di G3 (test set) | 4.51 |


### Classificazione: promosso o bocciato

Modello base: (Regressione Logistica, soglia di decisione 0.5):

| Metric | Score |
|---|---:|
| Accuracy | 0.66 |
| Precision (Bocciato) | 0.47 |
| Recall (Bocciato) | 0.35 |
| F1-score (Bocciato) | 0.40 |
| Precision (Promosso) | 0.72 |
| Recall (Promosso) | 0.81 |
| F1-score (Promosso) | 0.76 |



Modello ottimizzato: (Random Forest, `GridSearchCV` + soglia di decisione 0.65):

| Metric | Score |
|---|---:|
| Accuracy | 0.67 |
| Precision (Bocciato) | 0.50 |
| Recall (Bocciato) | 0.50 |
| F1-score (Bocciato) | 0.50 |
| Precision (Promosso) | 0.75 |
| Recall (Promosso) | 0.75 |
| F1-score (Promosso) | 0.75 |

Migliori iperparametri trovati: `n_estimators = 400`, `max_depth = 8` (accuratezza media in cross-validation: 70,6%).


### Effetto dei voti intermedi (controprova)

| Feature set | RMSE (regressione) |
|---|---:|
| Senza G1 / G2 | 3.91 |
| Con G1 / G2 | 2.01 |

## Key Insights

1. Il background e lo stile di vita spiegano solo una parte del voto finale. Un RMSE di 3.91 contro una deviazione standard di G3 di 4.51 indica che il modello riduce l'errore rispetto a una previsione ingenua basata sulla media, ma non di moltissimo: una quota consistente della variabilità del voto dipende da fattori non presenti nel dataset (es. impegno effettivo o difficoltà delle materie).

2. Le assenze e le bocciature pregresse sono assolutamente le variabili più predittive, ma il loro ordine di importanza si inverte a seconda del task. Per la regressione contano di più le assenze, che hanno una scala ampia (fino a 75) e permettono al modello di affinare la previsione punto per punto. Per la classificazione contano di più le bocciature, che pur avendo solo 4 valori possibili offrono un confine di separazione molto netto tra promosso e bocciato.

3. Il modello base fatica a riconoscere gli studenti bocciati: un recall di 0.35 significa che poco più di uno studente bocciato su tre viene identificato correttamente, conseguenza diretta dello sbilanciamento delle classi (67% promossi). Il modello raggiunge un'accuratezza discreta, ma per un caso d'uso in cui l'obiettivo realistico è individuare gli studenti a rischio, il recall sulla classe "Bocciato" è la metrica più importante da migliorare.

4. L'ottimizzazione (tuning + soglia di decisione) migliora il modello su tutti i fronti, non solo su uno a scapito di un altro: il recall sulla classe Bocciato raddoppia (0.35 → 0.50) e anche l'accuratezza generale sale leggermente (66% → 67%). Questo perché la soglia di default (0.5) era già distorta a favore della classe maggioritaria a causa dello sbilanciamento, e correggerla non comporta un vero compromesso.

5. I voti intermedi (G1, G2), se disponibili, restano la fonte di informazione più potente in assoluto: la loro reintroduzione dimezza quasi l'errore di regressione (RMSE da 3.91 a 2.01), confermando che il rendimento pregresso nello stesso anno scolastico è un predittore molto più diretto del voto finale rispetto a qualunque combinazione di variabili socio-demografiche.



##Dichiarazione su LLM
è stato utilizzato Claude per piccole correzioni e suggerimenti (ad esempio i suggerimenti futuri, se non quello sull'ampliamento del dataset, così come il reintegro delle variabili G1 e G2). Anche la presentazione è stata generata da Claude, utilizzata come una bozza poi modificata per avere una base estetica piacevole. 
