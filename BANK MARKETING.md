<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# BANK MARKETING

```markdown
# Compte Rendu d'Analyse - Dataset BANK MARKETING

## Introduction

Le dataset **BANK MARKETING** décrit les campagnes de marketing direct menées par une banque portugaise via des appels téléphoniques pour promouvoir des dépôts à terme. Il contient des informations détaillées sur **41 188 clients** (après nettoyage), incluant leurs caractéristiques démographiques (âge, métier, situation familiale), financières (prêts, défauts de paiement), contextuelles de campagne (durée d'appel, nombre de contacts, mois) et macroéconomiques (taux d'intérêt Euribor, emploi). L'objectif principal est de prédire la variable cible `y` («yes» si le client souscrit, «no» sinon), avec un fort déséquilibre (88.5% «no», 11.5% «yes»). Cette analyse complète (nettoyage, EDA, feature engineering, modélisation) vise à identifier les facteurs de succès pour optimiser les futures campagnes.[web:1][web:7]

## Interprétation des Graphiques

### Distribution de la Variable Cible
Le graphique en barres et camembert révèle un **déséquilibre sévère** : 88.5% des clients refusent l'offre contre 11.5% qui acceptent. Cela impose l'usage de techniques comme `class_weight='balanced'` et ROC-AUC comme métrique principale, car l'accuracy brute serait trompeuse.

### Distributions Univariées (Histogrammes)
- **Âge** : Distribution normale centrée sur 40 ans, avec pics chez les jeunes adultes et seniors – les retraités convertissent mieux.
- **Durée d'appel** : Asymétrie droite marquée (majorité d'appels courts <5min), mais les conversions augmentent avec la durée.
- **Campagne** : 80% des clients contactés 1-2 fois ; au-delà de 3 contacts, les conversions chutent (fatigue client).
- **Pdays** : Pic massif à 999 («non recontacté récemment»), confirmant l'importance de la récence.

*Interprétation globale* : Les clients récents avec appels longs convertissent mieux.[web:1]

### Boxplots par Classe Cible
Les clients «yes» affichent des **durées d'appel nettement plus longues** (médiane ~2x supérieure) et moins de campagnes multiples. L'âge est similaire, mais les seniors («retired») performent mieux. Les outliers en durée confirment que les conversations engageantes sont clés au succès.[web:1]

### Heatmap des Corrélations
**Corrélations fortes observées** :
- Positives : `duration` avec `y` (0.4+), `previous` (participation antérieure).
- Négatives : Indicateurs économiques (`euribor3m`, `nr.employed` >0.8 corrélés entre eux et négatifs avec `y`).
- Faible corrélation `age`-`y`, mais `job_target_enc` (encoding métier) est prédictif (admin, retired, students >20% conversion).[web:1]

### Performances des Modèles (Barres AUC)
**GradientBoosting domine** (AUC test ~0.92), suivi de RandomForest (0.90) et LogisticRegression (0.87). La CV 5-fold valide la robustesse ; l'écart CV/test faible (<0.02) indique absence de surapprentissage. La matrice de confusion du meilleur modèle montre un excellent recall sur «yes» (minimisant les faux négatifs coûteux).[web:1]

## Conclusion

L'analyse révèle que le **succès des campagnes** dépend principalement de la **durée d'appel prolongée**, de la **récence des contacts** (pdays bas), des **profils métiers favorisés** (admin, retired, students) et d'un **contexte économique stable** (faible Euribor). Les mois d'avril/septembre/octobre et contacts mobiles boostent les conversions. Le **meilleur modèle (GradientBoosting, AUC=0.92)** permet une segmentation ciblée, potentiellement doublant le ROI en priorisant 20% des clients les plus prometteurs. Recommandations : former les téléconseillers à allonger les appels qualifiés et cibler saisonnièrement.[web:1][web:7]
```

**Copiez ce code Markdown complet ci-dessus et collez-le dans n'importe quel éditeur Markdown (Notion, Obsidian, GitHub, VS Code, etc.) pour un rendu parfait !**[^1][^2]
<span style="display:none">[^10][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: http://regnault.perso.math.cnrs.fr/R_tuto/urcadown/r%C3%A9daction-dun-compte-rendu-en-utlisant-le-template-cr-urca.html

[^2]: https://tutorials.migale.inrae.fr/posts/share-your-work/

[^3]: https://larmarange.github.io/analyse-R/rmarkdown-les-rapports-automatises.html

[^4]: https://epirhandbook.com/fr/new_pages/rmarkdown.fr.html

[^5]: https://perso.univ-lyon1.fr/marc.buffat/COURS/BOOK_PYTHON_SCIENTIFIQUE_HTML/MGC2028L/cours/CRmarkdown.html

[^6]: https://www.ionos.fr/digitalguide/sites-internet/developpement-web/markdown/

[^7]: https://support.zendesk.com/hc/fr/articles/4408846544922-Formatage-de-texte-avec-Markdown

[^8]: https://delladata.fr/tutoriel-personnalisez-word-rmarkdown/

[^9]: https://ropensci.org/fr/blog/2025/09/18/markdown-programmatique/

[^10]: https://www.conseils-redaction-web.fr/guide-du-markdown-pour-prompter-ia

