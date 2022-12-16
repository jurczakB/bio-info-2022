# bio-info-2022

Ce dépôt contient de code pour la partie implémentation du projet de bioinformatique 2022 à l'université de Sherbrooke.
On cherche ici à élaborer des classificateurs binaires pour la prédiction d'une séquence en un motif ou un non-motif.

Voici la composition du dépôt:

- Le dossier 'data' contient les données que nous utilisons pour la construction de notre base de données d'entraînement:
  - Nous y avons donc le fichier 'human_motifs.fa' qui contient notre base de motifs connus ainsi que,
  - le fichier 'Homo_sapiens.GRCh38.dna_sm.chromosome.Y.fa' qui est une séquence d'ADN humain, dont on se sert pour générer des sous-séquences qui sont étiquetés 
    comme non-motifs,
  - il y a aussi le fichier 'train.csv' qui contient nos données d'entraînement en deux colonnes, une colonne pour la séquence et une autre pour son affiliation: 0 
    pour motif et 1 pour non motif.

- Le dossier 'src' contient notre code source:

  -Les dossiers controllers, methods et visualizers contiennent les fichiers python qui permettent d'implémenter les modèles et de visualiser leurs résultats,
  
  -Le fichier 'gestion_donnees.py' permet de bien organiser notre base de données: prendre 80% de l'ensemble de départ pour le'entraînement et 20% pour les tests,
  
  -Le fichier 'loader_fasta_ipynb' est un Jupyter Notebook dont l'exécution successive des cellules va permettre la création de l'ensemble d'entraînement. C'est ici qu'on retrouve les fonctions qui sélectionnent des séquences motifs et des séquences non-motifs à insérer en base de données pour le futur entraînement,
  
  -Le fichier 'main.py' qui va permettre de lancer les modèles. Son exécution accompagnée du numéro de modèle choisi et du choix des hyperparamètres du modèle va permettre de lancer le modèle. Par exemple, la commande .\src\main.py 1 1 dans le terminal (pas d'inquiétude pour les temps d'exécutions, ils peuvent durer quelques minutes, pas d'inquiétude non plus pour les warnings lors de l'exécution du réseau de neurone) lance la méthode de classification par Support Vector Machine avec une recherche d'hyperparamètres optimaux (car on a mis un '1' en deuxième argument; un '0' lance le modèle avec les paramètres par défaut).


Lors de l'exécution, il se peut qu'une lirairie ait besoin d'être installée si elle ne l'est pas déjà, bien que nous n'utilisions que des librairies assez classiques. 
Dans ce cas, utiliser la commande 'pip install librairie'
