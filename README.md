# Projet de session, GIF-1005: Introduction à l'apprentissage automatique
## Session automne 2025

# Projet: Prédiction du chaos par apprentissage machine
Le but de ce projet est d'effectuer la prédiction de séries temporelles chaotiques utilisant différentes approches d'apprentissage automatique et de comparer les performances. Les approches utilisées sont:
1. Calcul par réservoir (*reservoir computing* en anglais);
2. Réseau de neuronnes récurrent LSTM
3. Mécanisme récurrents indépendants (*recurrent independent mechanisms* en anglais)

L'attracteur de Lorenz sera étudié, permettant d'obtenir trois séries temporelles dépendantes à l'aide d'un système d'équations différentielles non-linéaires couplées. Le système de Lorenz est:

$$\frac{\text{d}x}{\text{d}t} = \sigma(y - x)$$

$$\frac{\text{d}y}{\text{d}t} = x(\rho - z) - y$$

$$\frac{\text{d}y}{\text{d}t} = xy - \beta z$$

Comprendre exactement ce que $x$, $y$ et $z$ représentent n'est pas nécessaire dans ce projet. Il faut toutefois savoir que ces équations représentent un modèle simpliste de convection atmosphérique et que $\sigma$, $\rho$ et $\beta$ sont des constantes, historiquement fixées à $10$, $28$ et $\frac{8}{3}$ respectivement[^1].

## Pourquoi étudier la prédiction du chaos? Pourquoi l'apprentissage automatique?
Un système chaotique est un système grandement sensible aux conditions initiales, ce qui fait que, sur une certaine période de temps, deux systèmes proches, mais légèrement différents, vont diverger. Plusieurs systèmes réels sont chaotiques, par exemple un pendule double ou encore certains cas du problème à $N$ corps. On entend parfois parler de l'effet papillon, soit *est-ce qu'un papillon battant des ailes au Brésil peut causer une tornade au Texas?*, ce qui illustre la sensibilité aux perturbations des systèmes complexes et chaotiques. La prédictions de dynamiques chaotiques, comme la météorologie, repose sur des modèles souvent très complexes en interprétabilité et qui dépendent des données du passé. Or, ces modèles peuvent être appelés à changer, notamment avec les changements climatiques, d'où l'intérêt de l'apprentissage automatique.

L'apprentissage automatique moderne repose en grande partie sur une approche non paramétrique de l'analyse des données: à partir des données, on veut extraire les relations importantes sans nécessairement savoir de quelles natures elles sont. Par exemple, on peut s'intéresser à la détection automatique de visage et on laisse l'algorithme de reconnaissance extraire lui-même les propriétés importantes qui composent un visage à l'aide de plusieurs exemples, autant variés que possible. Ainsi, l'algorithme définie son propre modèle de visages, sans que l'humain n'ait à définir ce qu'est un visage, tâche qui serait difficile. Naturellement l'humain reconnaît les visage facilement, mais comment transmettre cette faculté à une machine est une tâche difficile qui ne fait qu'ajouter un problème de plus.

En ayant cette mentalité, on peut s'intéresser aux systèmes chaotiques. Humainement, ces systèmes sont difficiles à interpréter et décrire, mais l'important est dans les données. En ayant le passé, on veut prédire le futur sans avoir le modèle exact; la plupart des systèmes réels ont un modèle théorique inconnu ou incomplet. Une approche basée sur les données est donc très attrayante. L'attracteur de Lorenz est un cas typique de système chaotique intéressant: il modélise une réalité (bien que simpliste) et on connaît le modèle. On peut alors évaluer les approches d'apprentissage machine quant à la prédiction d'un système chaotique. Si ces approches s'avèrent utiles avec Lorenz, pourquoi ne pas les utiliser avec d'autres modèles? 

# TODO: À finir: expliquer les approches utilisées (RC, RNN, RIMs)
# TODO: Est-ce que le texte est bon? Choses à changer?

[^1]: E. N. Lorenz, « Deterministic Nonperiodic Flow », en, Journal of the Atmospheric Sciences 20, 130-141 (1963).
