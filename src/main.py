# - Library -
import pandas as pd

# - DataBase Controller -
import sys

import gestion_donnees as gd

# - Controllers -
import controllers.svm_classifier_controller as svmc
import controllers.LogReg_classifier_controller as lrcc
import controllers.neural_network_classifier_controller as nncc

# - Visualizers -
import visualizers.classifierShowdown_Visualizer as cSV
import visualizers.learning_curve as lcV

from sklearn.metrics import log_loss


def main():

    usage = " \n Usage : python .\src\main.py method search_HyperParameters (Classifiers Showdown)\
    \n\t method : 1 => Support Vector Classification\
    \n\t method : 2 => Logistic Regression Classification\
    \n\t method : 3 => Neural Network Classification\
    \n\n\t search_HyperParameters : 0 => Default HyperParameters\
    \n\t search_HyperParameters : 1 => Search HyperParameters\
    \n\n\t Classifiers Showdown : 1 => Make the comparaison between all the model tuned without their hyperparameters\
    \n\t Classifiers Showdown : 2 => Make the comparaison between all the model tuned WITH their hyperparameters"

    if len(sys.argv) == 0 or len(sys.argv) >= 4:
        print(usage)
        return

     # - Gestion Data -
    gestion_donnees = gd.GestionDonnees()
    gestion_donnees.prepocess()
    gestion_donnees.stratifiedSelection()
    x_train, y_train, x_test, y_test = gestion_donnees.x_train, gestion_donnees.y_train, gestion_donnees.x_test, gestion_donnees.y_test

    if len(sys.argv) == 3:
        method = sys.argv[1]
        search_HP = sys.argv[2]
        if search_HP == "0":
            search_HP = False
        elif search_HP == "1":
            search_HP = True

        print("Méthode sélectionnée : ")
        if method == "1":
            print("\t- Support Vector Classifier")
            controller = svmc.Svm_Classifier_Controller(
                search_HP, x_train, y_train)
        elif method == "2":
            print("\t- Logistic Regression Classifier")
            controller = lrcc.LogReg_Classifier_Controller(
                search_HP, x_train, y_train, x_test, y_test)
        elif method == "3":
            print("\t- Neural Network Classifier")
            controller = nncc.Neural_Network_Classifier_Controller(
                search_HP, x_train, y_train)
        else:
            print(usage)
            return

        if (controller is None):
            print("\t- Undefined method")
            return
        else:
            classifier = controller.getClassifier()
            if search_HP:
                visualizer = controller.getVisualizer()

                print("Start : Visualisation du score en fonction des paramètres")
                visualizer.Visualise_tuning()
                print("End : Visualisation du score en fonction des paramètre")

        print("Start : Entrainement du modèle sur les paramètres donnés")
        classifier.train(x_train, y_train)
        print("End : Entrainement du modèle sur les paramètres donnés")

        score = classifier.scoreKfold(x_train, y_train)
        print('kfold score :')
        scores = gd.display_scores(score)
        print(scores)

        if method == "2": 
            print("Start : Visualisation des pénalties L1 et L2")
            visualizer = controller.getVisualizer()
            visualizer.Visualise_penalty()
            print("End : Visualisation des pénalties L1 et L2")


        logloss = classifier.logloss(x_train, y_train)
        print('Logloss score sur les données d"entrainement : ', logloss)

        print("Start : Visualisation de l'apprentissage du modèle")
        title = classifier.__class__.__name__
        lcV.learn_curve.plot_learning_curve(
            classifier.model, title, x_train, y_train, cv=2, scoring="accuracy").show()
        print("End : Visualisation de l'apprentissage du modèle")


        accuracy = classifier.global_accuracy(x_test, y_test)
        print("Accuracy sur les données de test : ", accuracy)
        
        logloss = classifier.logloss(x_test, y_test)
        print('Logloss score sur les données de test : ', logloss)

    if len(sys.argv) == 2:
        showdown = sys.argv[1]
        if showdown == "1":
            search_HP = False
        if showdown == "2":
            search_HP = True

        print("Beginning Confrontation des classifieurs : ")
        results_df = gd.showdown_df()
        classifiers = [
            svmc.Svm_Classifier_Controller(
                search_HP, x_train, y_train).getClassifier(),
            lrcc.LogReg_Classifier_Controller(
                search_HP, x_train, y_train).getClassifier(),]

        for clf in classifiers:
            clf.train(x_train, y_train)
            name = clf.__class__.__name__

            # print("="*30)
            # print(name)

            # print('****Results****')

            acc = clf.global_accuracy(x_test, y_test)
            #print("Accuracy: {:.4%}".format(acc))

            # train_predictions = clf.predict_proba(x_test)
            ll = clf.logloss(x_test, y_test)
            # print("Log Loss: {}".format(ll))

            log_entry = gd.showdownPutter(name, acc, ll)

            #results_df = results_df.concat(log_entry)
            results_df = pd.concat([results_df, log_entry])
        print(results_df)
        print("Ending Confrontation des classifieurs : ")
        # cSV.accuracyPlotter(results_df)
        cSV.subPlotter121(results_df)


if __name__ == "__main__":
    main()
