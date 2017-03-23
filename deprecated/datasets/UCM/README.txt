======
README
======
  The files in this archive are MatLab implementations for the algorithms presented in the
IROS 2012 submission, Combining Classification and Regression for WiFi Localization of 
Heterogeneous Robot Teams in Unknown Environements, by Benjamin Balaguer, Gorkem Erinc, 
and Stefano Carpin. We strongly encourage potential users to read the paper to get 
a better understanding of the implemented algorithms, file structure, and provided data. 
Depending on the archive downloaded, the included files are for the raw data, the 
processed data, the trained data, the algorithm implementations, or all of them together. 
Depending on the downloaded archive, different files will be available.

  The files are essentially divided into three distinct datasets, named GorkemUCM, 
Kavraki, and RealRobot. The GorkemUCM dataset was manually taken in the first floor 
of the engineering building at the University of California, Merced. It is comprised 
of 156 locations, each with 20 WiFi readings per location. The Kavraki dataset is 
publicly available at http://www.cis.upenn.edu/~ahae/~rwl-toolkit/ and was used for 
the paper: On the Feasibility of Using Wireless Ethernet for Indoor Localization, by 
Andrew Ladd, Kostas Bekris, Algis Rudys, Dan Wallach, and Lydia Kavraki. It is comprised 
of 3 floors of the computer science building on the Rice university campus, totaling 
506 locations, each with 100 readings per location. The RealRobot dataset is comprised 
of 10 robot runs in the first floor of the engineering building at the University of 
California, Merced. It is comprised of 110 locations, each with 10 readings per location.
We note that the GorkemUCM and RealRobot datasets are recorded in WiFi signal strengths, 
whereas the Kavraki dataset is recorded in signal-to-interferance ratios.

File Structure:
---------------
  Each dataset has its own directory, named GorkemUCM, Kavraki, RealRobot. Each 
of the dataset directories have an algorithm directory (DecisionTree, GaussianMixtureBen, 
GaussianMixtureKavraki, LRoneVall, MultinomialLogit, NearestNeighbor, RandomForest, 
SVMoneVall), as well as a training directory (TrainingRandomized). The algorithm 
directory stores the datastructure produced by the algorithm, after it has been trained. 
The training directory stores processed versions of the dataset, which separates training 
data from classification data based on the number of readings per locations used during 
training, |s|, and randomized 50 times. The main directory, UCM-WiFi-Localizer includes 
the raw data, stored as mat files, functions for each classification algorithm, regression 
algorithm, and monte carlo localization (particle filter).

  Raw Data:
  ---------
    Gorkem_data.mat: unmodified/unprocessed data for the GorkemUCM dataset.
    Kavraki_data.mat: unmodified/unprocessed data for the Kavraki dataset.
    RealRobot/*.txt: raw data files taken directly from the robot for the RealRobot dataset.    
    RealRobot_data.mat: raw data from the RealRobot/*.txt, formatted to be easier to process.
  
  Processed Data:
  ---------------
    -RealRobot_time.mat
      Timestamps for which the robot started acquiring signal strengths readings for the 
      RealRobot dataset.
    -GorkemUCM\TrainingRandomized\Trained_%s_%i.mat
      Processed GorkemUCM datasets where some of the data was used for training and the 
      rest for classifying. The %s indicates the percentage of data used for training and 
      the %i indicates one of the random samples, from 1 to 50.
    -Kavraki\TrainingRandomized\Trained_%s_%i.mat
      Processed Kavraki datasets where some of the data was used for training and the 
      rest for classifying. The %s indicates the percentage of data used for training and 
      the %i indicates one of the random samples, from 1 to 50.
    -RealRobot\Runs\%i.mat
      One of the real robot runs, where %i is a number between 1 and 10.

  Trained Data:
  -------------
    -%DataSet/%Algorithm/trained_%s_%i.mat
      Stores the data structure constructed by training %Algorithm (i.e. DecisionTree, 
      GaussianMixtureBen, GaussianMixtureKavraki, LRoneVall, MultinomialLogit, 
      NearestNeighbor, RandomForest, SVMoneVall) on the processed data set trained_%s_%i.
      %Dataset is either GorkemUCM or Kavraki, %s indicates the percentage of data used
      for training, and %i indicates one of the random samples, from 1 to 50. The data 
      structure that is part of the mat file is algorithm-dependent. For example, it 
      will be a decision tree for the DecisionTree algorithm, and regression coefficients 
      for Logsistic Regression (LRoneVall).
       
    -%DataSet/%Algorithm/Results.mat
      Resulting classification accuracy for the %Dataset and %Algorithm. The accuracy 
      is acquired by using the trained data and classifying the remaining data (the one 
      not used to train).

  Code:
  -----
    -%Algorithm.m
      Algorithm implementation, which consists of a training function and a classification
      function. The GorkemUCM and Kavraki datasets are processed differently since the 
      Kavraki dataset has 3 floors instead of 1 for the GorkemUCM dataset. The training 
      functions create the %DataSet/%Algorithm/trained_%s_%i.mat files, whereas the 
      classification functions create the %DataSet/%Algorithm/Results.mat files (see above).
    -ParticleFilter.m
      Implementation of the Monte Carlo Localization (aka particle filter). Please read the 
      paper for more information.
    -RandomizeTraining_%DataSet.m
      Code that reads the raw data for a specific %DataSet, processes it, and saves the 
      appropriate mat files to be used for training. These functions essentially takes the 
      Raw Data and outputs the Processed Data.
    -RegressionTest_RandomForest.m
      Various regression implementations, all based on results from the Random Forest 
      classification. Please read the paper for more information. 

Troubleshooting
---------------
  1) Code complains about not being able to open certain files.
        A lot of the paths are hardcoded, relative to the main directory. As such, it is 
        essential that the path structure encompassed in the zip files is preserved on
        your machine. In addition, you should make sure that the main directory (and 
        subdirectories) are added to your MatLab path.
  2) The SVM code does not work.
        We use an external SVM toolbox, that needs to be independently installed from
        http://asi.insa-rouen.fr/enseignants/~arakotom/toolbox. Once installed, make 
        sure the main directory (and sub-directories) are included in MatLab's path.
  3) MatLab complains about one of the functions being called in the code.
        If the problematic function is not included as part of our code base, it means
        that it came with our MatLab installation, unless it is specific to the SVM 
        code (see Troubleshooting 2 in that case). Our code runs on MatLab R2011b, 
        with all the toolboxes installed. It is possible that you are missing some 
        toolboxes.
  4) Something else not mentioned in this section.
        Please contact Benjamin Balaguer at bbalaguer@ucmerced.edu.